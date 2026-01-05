"""  Serves as the retriever for the EchoSight.
"""

import math
import os
import re
import torch
import torch.nn as nn
import tqdm
import pickle
import json
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)
import faiss
import numpy as np
from faiss import write_index, read_index
import faiss.contrib.torch_utils


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims, dropout: float = 0.1):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if not hidden_dims:
            raise ValueError("MLPProjector requires at least one hidden dimension.")
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EmbeddingProjector(MLPProjector):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 4096, dropout: float = 0.1):
        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dims=[hidden_dim], dropout=dropout)


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25,
                 decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.register_buffer("embedding", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))

    def forward(self, inputs: torch.Tensor):
        distances = (
            torch.sum(inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding ** 2, dim=1)
            - 2 * torch.matmul(inputs, self.embedding.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self.num_embeddings,
            device=inputs.device,
            dtype=self.embedding.dtype,
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding)

        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )
            dw = torch.matmul(encodings.t(), inputs)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
            self.embedding = self.ema_w / self.ema_cluster_size.unsqueeze(1)

        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, loss, perplexity


class VQVAEProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        num_embeddings: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.vq = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=output_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        quantized, _, _ = self.vq(z)
        return self.decoder(quantized)


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class DiffusionProjector(nn.Module):
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        *,
        time_dim: int = 256,
        hidden_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
        )
        input_total_dim = teacher_dim + student_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(input_total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp[0](t)
        t_emb = t_emb.to(condition.dtype)
        t_emb = self.time_mlp[1](t_emb)
        t_emb = self.time_mlp[2](t_emb)
        x_input = torch.cat([x, condition, t_emb], dim=1)
        return self.net(x_input)


class DiffusionSamplerProjector(nn.Module):
    def __init__(
        self,
        denoiser: DiffusionProjector,
        *,
        timesteps: int = 1000,
        sample_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        eta: float = 0.0,
    ):
        super().__init__()
        if sample_steps <= 0:
            raise ValueError("sample_steps must be > 0")
        if timesteps <= 0:
            raise ValueError("timesteps must be > 0")
        if sample_steps > timesteps:
            raise ValueError("sample_steps cannot exceed timesteps")
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.sample_steps = int(sample_steps)
        self.eta = float(eta)

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        step_indices = torch.linspace(0, timesteps - 1, self.sample_steps)
        self.register_buffer("step_indices", step_indices.long())

    def forward(self, condition: torch.Tensor, *, sample_steps: int | None = None, eta: float | None = None) -> torch.Tensor:
        steps = self.sample_steps if sample_steps is None else int(sample_steps)
        if steps <= 0:
            raise ValueError("sample_steps must be > 0")
        if steps > self.timesteps:
            raise ValueError("sample_steps cannot exceed timesteps")
        eta_val = self.eta if eta is None else float(eta)

        if steps == self.sample_steps:
            step_indices = self.step_indices
        else:
            step_indices = torch.linspace(0, self.timesteps - 1, steps, device=condition.device).long()

        batch = condition.size(0)
        device = condition.device
        dtype = condition.dtype
        teacher_dim = self.denoiser.teacher_dim
        x = torch.randn(batch, teacher_dim, device=device, dtype=dtype)

        for i in reversed(range(len(step_indices))):
            t = int(step_indices[i].item())
            t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
            eps = self.denoiser(x, condition, t_batch)

            alpha_t = self.alphas_cumprod[t].to(dtype=x.dtype)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
            pred_x0 = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t

            if i == 0:
                x = pred_x0
                break

            t_prev = int(step_indices[i - 1].item())
            alpha_prev = self.alphas_cumprod[t_prev].to(dtype=x.dtype)
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = torch.sqrt(1.0 - alpha_prev)

            if eta_val > 0:
                sigma = eta_val * torch.sqrt(
                    (1.0 - alpha_prev) / (1.0 - alpha_t)
                ) * torch.sqrt(1.0 - alpha_t / alpha_prev)
                noise = torch.randn_like(x)
            else:
                sigma = torch.zeros((), device=device, dtype=dtype)
                noise = torch.zeros_like(x)

            x = sqrt_alpha_prev * pred_x0 + torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma ** 2, min=0.0)) * eps + sigma * noise

        return x


_MLP_WEIGHT_RE = re.compile(r"^net\.(\d+)\.weight$")


def _extract_state_dict(projector_path: str) -> dict:
    state = torch.load(projector_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state_dict = state["state_dict"]
    else:
        state_dict = state

    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid projector checkpoint format: {projector_path}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _infer_mlp_dims(state_dict: dict) -> tuple[int, list[int], int]:
    linear_weights = []
    for key, tensor in state_dict.items():
        match = _MLP_WEIGHT_RE.match(key)
        if match is None:
            continue
        if tensor.ndim != 2:
            continue
        linear_weights.append((int(match.group(1)), tensor))
    if len(linear_weights) < 2:
        raise KeyError("MLP projector checkpoint must contain at least 2 linear layer weights.")
    linear_weights.sort(key=lambda x: x[0])
    input_dim = int(linear_weights[0][1].shape[1])
    hidden_dims = [int(w.shape[0]) for _, w in linear_weights[:-1]]
    output_dim = int(linear_weights[-1][1].shape[0])
    prev_out = input_dim
    for _, w in linear_weights:
        in_dim = int(w.shape[1])
        if in_dim != prev_out:
            raise ValueError("MLP projector weight shapes are inconsistent.")
        prev_out = int(w.shape[0])
    return input_dim, hidden_dims, output_dim


def _build_mlp_projector(state_dict: dict, projector_kwargs: dict | None) -> MLPProjector:
    projector_kwargs = projector_kwargs or {}
    hidden_dims = projector_kwargs.get("hidden_dims", projector_kwargs.get("hidden_dim"))
    dropout = float(projector_kwargs.get("dropout", 0.1))
    input_dim, inferred_hidden_dims, output_dim = _infer_mlp_dims(state_dict)
    if hidden_dims is None:
        hidden_dims = inferred_hidden_dims
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    if list(hidden_dims) != inferred_hidden_dims:
        raise ValueError(
            f"Projector hidden dims {hidden_dims} do not match checkpoint {inferred_hidden_dims}."
        )
    return MLPProjector(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims, dropout=dropout)


def _load_mlp_projector(
    projector_path: str,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    projector_kwargs: dict | None = None,
) -> MLPProjector:
    if device.type != "cuda" and torch_dtype in (torch.float16, torch.bfloat16):
        torch_dtype = torch.float32
    state_dict = _extract_state_dict(projector_path)
    proj = _build_mlp_projector(state_dict, projector_kwargs)
    filtered = {k: v for k, v in state_dict.items() if k.startswith("net.")}
    incompatible = proj.load_state_dict(filtered, strict=False)
    if incompatible.missing_keys:
        raise KeyError(f"Projector checkpoint missing keys: {incompatible.missing_keys}")
    proj.to(device=device, dtype=torch_dtype)
    proj.eval()
    return proj


def _infer_vqvae_dims(state_dict: dict) -> tuple[int, int, int, int]:
    enc0 = state_dict.get("encoder.0.weight", None)
    enc3 = state_dict.get("encoder.3.weight", None)
    dec0 = state_dict.get("decoder.0.weight", None)
    vq_emb = state_dict.get("vq.embedding", None)
    if enc0 is None or enc3 is None or dec0 is None or vq_emb is None:
        raise KeyError("VQ-VAE projector checkpoint is missing required keys.")
    hidden_dim, input_dim = tuple(enc0.shape)
    output_dim, hidden_dim_2 = tuple(enc3.shape)
    if hidden_dim_2 != hidden_dim:
        raise ValueError("VQ-VAE encoder hidden dim mismatch.")
    hidden_dim_dec, output_dim_2 = tuple(dec0.shape)
    if hidden_dim_dec != hidden_dim or output_dim_2 != output_dim:
        raise ValueError("VQ-VAE decoder shape mismatch.")
    num_embeddings, embedding_dim = tuple(vq_emb.shape)
    if embedding_dim != output_dim:
        raise ValueError("VQ-VAE codebook dim mismatch.")
    return input_dim, hidden_dim, output_dim, num_embeddings


def _load_vqvae_projector(
    projector_path: str,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    projector_kwargs: dict | None = None,
) -> VQVAEProjector:
    if device.type != "cuda" and torch_dtype in (torch.float16, torch.bfloat16):
        torch_dtype = torch.float32
    state_dict = _extract_state_dict(projector_path)
    input_dim, hidden_dim, output_dim, num_embeddings = _infer_vqvae_dims(state_dict)
    commitment_cost = float((projector_kwargs or {}).get("commitment_cost", 0.25))
    decay = float((projector_kwargs or {}).get("decay", 0.99))
    proj = VQVAEProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_embeddings=num_embeddings,
        commitment_cost=commitment_cost,
        decay=decay,
    )
    proj.load_state_dict(state_dict, strict=True)
    proj.to(device=device, dtype=torch_dtype)
    proj.eval()
    return proj


def _infer_diffusion_dims(state_dict: dict) -> tuple[int, int, int, int]:
    time_w = state_dict.get("time_mlp.1.weight", None)
    if time_w is None:
        raise KeyError("Diffusion projector checkpoint is missing time_mlp.1.weight.")
    time_dim = int(time_w.shape[0])
    linear_weights = []
    for key, tensor in state_dict.items():
        match = _MLP_WEIGHT_RE.match(key)
        if match is None:
            continue
        linear_weights.append((int(match.group(1)), tensor))
    if not linear_weights:
        raise KeyError("Diffusion projector checkpoint is missing net.* weights.")
    linear_weights.sort(key=lambda x: x[0])
    input_total_dim = int(linear_weights[0][1].shape[1])
    hidden_dim = int(linear_weights[0][1].shape[0])
    teacher_dim = int(linear_weights[-1][1].shape[0])
    student_dim = input_total_dim - teacher_dim - time_dim
    if student_dim <= 0:
        raise ValueError("Diffusion projector dimensions are inconsistent.")
    return student_dim, teacher_dim, time_dim, hidden_dim


def _load_diffusion_projector(
    projector_path: str,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    projector_kwargs: dict | None = None,
) -> DiffusionSamplerProjector:
    if device.type != "cuda" and torch_dtype in (torch.float16, torch.bfloat16):
        torch_dtype = torch.float32
    state_dict = _extract_state_dict(projector_path)
    student_dim, teacher_dim, time_dim, hidden_dim = _infer_diffusion_dims(state_dict)
    proj = DiffusionProjector(
        student_dim=student_dim,
        teacher_dim=teacher_dim,
        time_dim=time_dim,
        hidden_dim=hidden_dim,
    )
    proj.load_state_dict(state_dict, strict=True)
    proj.to(device=device, dtype=torch_dtype)
    proj.eval()

    projector_kwargs = projector_kwargs or {}
    timesteps = int(projector_kwargs.get("diffusion_timesteps", 1000))
    sample_steps = int(projector_kwargs.get("diffusion_steps", 50))
    beta_start = float(projector_kwargs.get("diffusion_beta_start", 0.0001))
    beta_end = float(projector_kwargs.get("diffusion_beta_end", 0.02))
    eta = float(projector_kwargs.get("diffusion_eta", 0.0))
    sampler = DiffusionSamplerProjector(
        proj,
        timesteps=timesteps,
        sample_steps=sample_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        eta=eta,
    )
    sampler.to(device=device, dtype=torch_dtype)
    sampler.eval()
    return sampler


def _load_projector(
    projector_path: str,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    projector_type: str | None = None,
    projector_kwargs: dict | None = None,
) -> nn.Module:
    projector_type = (projector_type or "mlp2").strip().lower()
    if projector_type in {"mlp", "mlp2", "mlp3", "contrastive"}:
        return _load_mlp_projector(
            projector_path,
            device=device,
            torch_dtype=torch_dtype,
            projector_kwargs=projector_kwargs,
        )
    if projector_type in {"vqvae", "vq-vae"}:
        return _load_vqvae_projector(
            projector_path,
            device=device,
            torch_dtype=torch_dtype,
            projector_kwargs=projector_kwargs,
        )
    if projector_type in {"diffusion", "ddpm"}:
        return _load_diffusion_projector(
            projector_path,
            device=device,
            torch_dtype=torch_dtype,
            projector_kwargs=projector_kwargs,
        )
    raise ValueError(f"Unsupported projector_type: {projector_type}")


def _load_embedding_projector(
    projector_path: str,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    projector_type: str | None = None,
    projector_kwargs: dict | None = None,
) -> nn.Module:
    return _load_projector(
        projector_path,
        device=device,
        torch_dtype=torch_dtype,
        projector_type=projector_type,
        projector_kwargs=projector_kwargs,
    )


class KnowledgeBase:
    """Knowledge base for EchoSight system.

    Returns:
        KnowledgeBase
    """

    def __len__(self):
        """Return the length of the knowledge base.

        Args:

        Returns:
            int
        """
        return len(self.knowledge_base)

    def __getitem__(self, index):
        """Return the knowledge base entry at the given index.

        Args:
            index (int): The index of the knowledge base entry to return.

        Returns:
            KnowledgeBaseEntry
        """
        return self.knowledge_base[index]

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = None

    def load_knowledge_base(self):
        """Load the knowledge base."""
        raise NotImplementedError


class WikipediaKnowledgeBase(KnowledgeBase):
    """Knowledge base for EchoSight."""

    def __init__(self, knowledge_base_path):
        """Initialize the KnowledgeBase class.

        Args:
            knowledge_base_path (str): The path to the knowledge base.
        """
        super().__init__(knowledge_base_path)
        self.knowledge_base = []

    def load_knowledge_base_full(
        self, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base from multiple score files.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The parent folder path to the vision similarity scores to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None

        if visual_attr is not None:
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if scores_path is not None:
            # get the image scores for each entry
            # get all the *.pkl files in the scores_path
            print("Loading knowledge base score from {}.".format(scores_path))
            import glob

            score_files = glob.glob(scores_path + "/*.pkl")
            image_scores = {}
            for score_file in tqdm.tqdm(score_files):
                try:
                    with open(score_file, "rb") as f:
                        image_scores.update(pickle.load(f))
                except:
                    raise FileNotFoundError(
                        "Image scores not found, which should be a url or path to a pickle file."
                    )
            print("Loaded {} image scores.".format(len(image_scores)))
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            print("Loading knowledge base without image scores.")
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base

    def load_knowledge_base(self, image_dict=None, scores_path=None, visual_attr=None):
        """Load the knowledge base.

        Args:
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
        """
        try:
            with open(self.knowledge_base_path, "rb") as f:
                knowledge_base_dict = json.load(f)
        except:
            raise FileNotFoundError(
                "Knowledge base not found, which should be a url or path to a json file."
            )
        # image_dict and load_scores_path can't be both None and can't be both not None
        if visual_attr is not None:
            # raise NotImplementedError
            try:
                with open(visual_attr, "r") as f:
                    visual_attr_dict = json.load(f)
            except:
                raise FileNotFoundError(
                    "Visual Attr not found, which should be a url or path to a json file."
                )

        if (
            scores_path is not None
        ):  # TODO: fix the knowledge base and visual_attr is None:
            # get the image scores for each entry
            print("Loading knowledge base score from {}.".format(scores_path))
            try:
                with open(scores_path, "rb") as f:
                    image_scores = pickle.load(f)
            except:
                raise FileNotFoundError(
                    "Image scores not found, which should be a url or path to a pickle file."
                )
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                for url in wiki_entry.image_urls:
                    if url in image_scores:
                        wiki_entry.score[url] = image_scores[url]
                self.knowledge_base.append(wiki_entry)
        else:
            print("Loading knowledge base without image scores.")
            for wiki_url, entry in knowledge_base_dict.items():
                wiki_entry = WikipediaKnowledgeBaseEntry(entry)
                self.knowledge_base.append(wiki_entry)

        print("Loaded knowledge base with {} entries.".format(len(self.knowledge_base)))
        return self.knowledge_base


class WikipediaKnowledgeBaseEntry:
    """Knowledge base entry for EchoSight.

    Returns:
    """

    def __init__(self, entry_dict, visual_attr=None):
        """Initialize the KnowledgeBaseEntry class.

        Args:
            entry_dict: The dictionary containing the knowledge base entry.
            visual_attr: The visual attribute. Deprecated in the current version.

        Returns:
            KnowledgeBaseEntry
        """
        self.title = entry_dict["title"]
        self.url = entry_dict["url"]
        self.image_urls = entry_dict["image_urls"]
        self.image_reference_descriptions = entry_dict["image_reference_descriptions"]
        self.image_section_indices = entry_dict["image_section_indices"]
        self.section_titles = entry_dict["section_titles"]
        self.section_texts = entry_dict["section_texts"]
        self.image = {}
        self.score = {}
        self.visual_attr = visual_attr


class Retriever:
    """Retriever parent class for EchoSight."""

    def __init__(self, model=None):
        """Initialize the Retriever class.

        Args:
            model: The model to use for retrieval.
        """
        self.model = model

    def load_knowledge_base(self, knowledge_base_path):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        raise NotImplementedError

    def retrieve_image(self, image):
        """Retrieve the image.

        Args:
            image: The image to retrieve.
        """
        raise NotImplementedError


class ClipRetriever(Retriever):
    """Image Retriever with CLIP-based VIT."""

    def __init__(
        self,
        model: str = "clip",
        device: str = "cpu",
        *,
        torch_dtype: torch.dtype = torch.float16,
        model_name: str | None = None,
        student_model_name: str | None = None,
        projector_path: str | None = None,
        projector_type: str | None = None,
        projector_kwargs: dict | None = None,
    ):
        """Initialize the ClipRetriever class.

        Args:
            model: The model to use for retrieval. Should be 'clip' or 'eva-clip'.
            device: The device to use for retrieval.
        """
        super().__init__(model)
        self.model_type = str(model)
        self.torch_dtype = torch_dtype
        self.projector = None
        if model == "clip":
            self.model = AutoModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=self.torch_dtype,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        elif model == "eva-clip":
            eva_name = model_name or "BAAI/EVA-CLIP-8B"
            self.model = AutoModel.from_pretrained(
                eva_name,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
            )
            del self.model.text_projection
            del self.model.text_model  # avoiding OOM
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.model.eval()
            self.processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        elif model in {"clip-mlp", "clip_mlp"}:
            student_name = student_model_name or "openai/clip-vit-large-patch14-336"
            if projector_path is None:
                raise ValueError("projector_path is required for model='clip-mlp'")
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                student_name,
                torch_dtype=self.torch_dtype,
            )
            self.processor = CLIPImageProcessor.from_pretrained(student_name)
            self.projector = _load_embedding_projector(
                projector_path,
                device=torch.device(device if torch.cuda.is_available() else "cpu"),
                torch_dtype=self.torch_dtype,
                projector_type=projector_type,
                projector_kwargs=projector_kwargs,
            )
            self.projector_type = projector_type
            self.projector_kwargs = projector_kwargs or {}
        self.device = device
        self.model.to(device)
        self.knowledge_base = None

    def load_knowledge_base(
        self, knowledge_base_path, image_dict=None, scores_path=None, visual_attr=None
    ):
        """Load the knowledge base.

        Args:
            knowledge_base_path: The knowledge base to load.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        self.knowledge_base.load_knowledge_base(
            image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
        )
        # if scores_path is a folder, then load all the scores in the folder, otherwise, load the single score file

    def save_knowledge_base_faiss(
        self,
        knowledge_base_path,
        image_dict=None,
        scores_path=None,
        visual_attr=None,
        save_path=None,
    ):
        """Save the knowledge base with faiss index.

        Args:
            knowledge_base_path: The knowledge base to load.
            image_dict: The image dictionary to load.
            scores_path: The path to the vision similarity score file to load.
            visual_attr: The visual attribute dictionary to load. Deprecated in the current version.
            save_path: The path to save the faiss index.
        """
        self.knowledge_base = WikipediaKnowledgeBase(knowledge_base_path)
        if scores_path[-4:] == ".pkl":
            print("Loading knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        else:
            print("Loading full knowledge base from {}.".format(scores_path))
            self.knowledge_base.load_knowledge_base_full(
                image_dict=image_dict, scores_path=scores_path, visual_attr=visual_attr
            )
        self.prepare_faiss_index()
        self.save_faiss_index(save_path)

    def retrieve_image(
        self, image, top_k=100, pool_method="max", return_entry_list=False
    ):
        raise NotImplementedError("Pleas use retrieve_image_faiss or retrieve_image_faiss_batch.")
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        inputs.to(self.device)
        outputs = self.model(**inputs)
        image_score = outputs.pooler_output
        # get the top k images in kb by cosine similarity
        kb_image_similarities = {}
        for i in range(len(self.knowledge_base)):
            kb_image_similarity = []
            wiki_url = self.knowledge_base[i].url
            image_urls = list(self.knowledge_base[i].score.keys())
            scores = [
                torch.tensor(self.knowledge_base[i].score[url]).to(self.device)
                for url in image_urls
            ]
            if len(scores) == 0:
                continue
            scores_matrix = torch.stack(scores, dim=0)
            kb_image_similarity = torch.cosine_similarity(
                image_score.unsqueeze(0), scores_matrix, dim=-1
            ).squeeze(0)
            if pool_method == "max":
                # get the max score
                # kb_image_similarity = torch.max(kb_image_similarity, dim=0)[0]
                # get the max score's index in the url list
                max_similarity_index = torch.argmax(kb_image_similarity, dim=0)
                max_similarity = kb_image_similarity[max_similarity_index]
                max_similarity_url = image_urls[max_similarity_index]
            else:
                raise NotImplementedError("Only max pooling is implemented.")
            # add key to the dict
            if wiki_url not in kb_image_similarities:
                kb_image_similarities[wiki_url] = {}
            kb_image_similarities[wiki_url].update(
                {"similarity": max_similarity.item()}
            )
            kb_image_similarities[wiki_url].update({"knowledge_base_index": i})
            kb_image_similarities[wiki_url].update(
                {"image_url": max_similarity_url}
            )  # TODO bug to fix, if multiple images of same entry are hit

        ranked_list = sorted(
            kb_image_similarities.items(),
            key=lambda x: x[1]["similarity"],
            reverse=True,
        )
        # get the top k images' urls
        top_k_entries = []
        if return_entry_list:
            for i in range(top_k):
                top_k_entries.append(
                    self.knowledge_base[ranked_list[i][1]["knowledge_base_index"]]
                )
            return top_k_entries
        for i in range(top_k):
            top_k_entries.append(
                {
                    "url": ranked_list[i][0],
                    "knowledge_base_index": ranked_list[i][1]["knowledge_base_index"],
                    "image_url": ranked_list[i][1]["image_url"],
                    "similarity": ranked_list[i][1]["similarity"],
                    "kb_entry": self.knowledge_base[
                        ranked_list[i][1]["knowledge_base_index"]
                    ],
                }
            )

        return top_k_entries

    def save_faiss_index(self, save_index_path):
        """Save the faiss index.
        
        Args:
            save_index_path: The path to save the faiss index.
        """
        if save_index_path is not None:
            write_index(self.faiss_index, save_index_path + "kb_index.faiss")
            with open(os.path.join(save_index_path, "kb_index_ids.pkl"), "wb") as f:
                pickle.dump(self.faiss_index_ids, f)

    # def load_faiss_index(self, load_index_path):
    #     """Load the faiss index.
        
    #     Args:
    #         load_index_path: The path to load the faiss index.
    #     """
    #     if load_index_path is not None:
    #         # Always load the CPU index first
    #         self.faiss_index = read_index(
    #             os.path.join(load_index_path, "kb_index.faiss")
    #         )

    #         # Load index id mapping with robust path join (no reliance on trailing slash)
    #         ids_pkl_path = os.path.join(load_index_path, "kb_index_ids.pkl")
    #         with open(ids_pkl_path, "rb") as f:
    #             self.faiss_index_ids = pickle.load(f)

    #         # Try to move to GPU when available; otherwise, keep CPU index
    #         moved_to_gpu = False
    #         try:
    #             if hasattr(faiss, "StandardGpuResources"):
    #                 res = faiss.StandardGpuResources()
    #                 self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
    #                 moved_to_gpu = True
    #         except Exception as e:
    #             print(
    #                 f"[FAISS] GPU acceleration unavailable or failed ({e}). Using CPU index instead."
    #             )

    #         loc = "GPU" if moved_to_gpu else "CPU"
    #         print(
    #             f"Faiss index loaded on {loc} with {self.faiss_index.ntotal} entries."
    #         )
    #     return
    
    def load_faiss_index(self, load_index_path, gpu_id: int = 0):
        """Load the faiss index with optional GPU transfer.

        Args:
            load_index_path (str): Path containing kb_index.faiss and kb_index_ids.pkl.
            gpu_id (int): GPU device ID to load the index onto (default: 0).
        """
        index_path = os.path.join(load_index_path, "kb_index.faiss")
        ids_path = os.path.join(load_index_path, "kb_index_ids.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"[FAISS] Index file not found: {index_path}")
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"[FAISS] ID file not found: {ids_path}")

        print(f"[FAISS] Loading index from {index_path}")
        self.faiss_index = read_index(index_path)
        with open(ids_path, "rb") as f:
            self.faiss_index_ids = pickle.load(f)
        print(f"[FAISS] CPU index loaded with {self.faiss_index.ntotal} entries.")

        # GPU 사용 여부 확인
        if not torch.cuda.is_available():
            print("[FAISS] CUDA not available → keeping index on CPU.")
            return

        try:
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
            est_size = self.faiss_index.ntotal * self.faiss_index.d * 4
            print(f"[FAISS] GPU {gpu_id}: {free_mem/1e9:.2f} GB free / {total_mem/1e9:.2f} GB total")
            print(f"[FAISS] Estimated index size: {est_size/1e9:.2f} GB")

            if est_size > free_mem * 0.8:
                print(f"[FAISS] ⚠️ Not enough VRAM on GPU:{gpu_id} → keeping on CPU.")
                return

            res = faiss.StandardGpuResources()
            print(f"[FAISS] Moving index to GPU:{gpu_id} ...")
            self.faiss_index = faiss.index_cpu_to_gpu(res, gpu_id, self.faiss_index)
            print(f"[FAISS] ✅ Index successfully loaded on GPU:{gpu_id}")

        except Exception as e:
            print(f"[FAISS] ⚠️ GPU load failed ({e}), fallback to CPU.")

        return


    def prepare_faiss_index(self):
        """Prepare the faiss index from scores in the knowledge base."""
        # use the knowledge base's score element to build the index
        # get the image scores for each entry
        scores = [
            score for entry in self.knowledge_base for score in entry.score.values()
        ]
        score_ids = [
            i
            for i in range(len(self.knowledge_base))
            for j in range(len(self.knowledge_base[i].score))
        ]
        # import ipdb; ipdb.set_trace()
        index = faiss.IndexFlatIP(scores[0].shape[0])
        # res = faiss.StandardGpuResources()
        # index = faiss.index_cpu_to_gpu(res, 0, index)
        np_scores = np.array(scores)
        np_scores = np_scores.astype(np.float32)
        faiss.normalize_L2(np_scores)
        index.add(np_scores)
        self.faiss_index = index
        self.faiss_index_ids = score_ids
        print("Faiss index built with {} entries.".format(index.ntotal))

        return


    def built_text_embedding(self, text_faiss_path):
        """Build the text mathcing faiss index from the knowledge base.
        
        Score is calculated by cosine similarity between the image and article text embeddings.
        
        Args:
            text_faiss_path: The path to save the text faiss index.
        """
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        kb_text = []
        for entry in self.knowledge_base:
            text = entry.title 
            for section in entry.section_texts:
                text += "\n" + section 
                break# only use the first section
            kb_text.append(text)
        inputs = tokenizer(kb_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        batch_size = 512
        outputs = []
        for i in range(0, len(kb_text), batch_size):
            text_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
            output = self.model.get_text_features(**text_inputs)
            outputs.extend(output.cpu().detach().numpy())
        # build the faiss index
        index = faiss.IndexFlatIP(outputs[0].shape[0])
        np_outputs = np.array(outputs)
        np_outputs = np_outputs.astype(np.float32)
        faiss.normalize_L2(np_outputs)
        index.add(np_outputs)
        self.faiss_index = index
        self.faiss_index_ids = [i for i in range(len(kb_text))]
        self.save_faiss_index(text_faiss_path)
        return
    
    @torch.no_grad()
    def retrieve_image_faiss(
        self, image, top_k=100, pool_method="max", return_entry_list=False
    ):
        """Retrieve the top K similar images from the knowledge base using faiss.

        Args:
            image: The image to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.
        """
        if self.model_type == "clip":
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.get_image_features(inputs)
        elif self.model_type == "eva-clip":
            # EVA-CLIP Process the input image
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_score = self.model.encode_image(inputs)
        elif self.model_type in {"clip-mlp", "clip_mlp"}:
            if self.projector is None:
                raise RuntimeError("clip-mlp retriever requires a loaded projector.")
            inputs = (
                self.processor(images=image, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            if hasattr(self.model, "get_image_features"):
                student = self.model.get_image_features(pixel_values=inputs)
            else:
                student = self.model(pixel_values=inputs).image_embeds
            student = student / student.norm(dim=-1, keepdim=True)
            image_score = self.projector(student)
        assert self.faiss_index and self.faiss_index_ids is not None
        query = image_score.float()
        query = torch.nn.functional.normalize(query)
        # Use CPU numpy for FAISS search to support CPU index
        query_np = query.detach().cpu().contiguous().numpy().astype("float32")
        D, I = self.faiss_index.search(query_np, top_k)
        top_k_entries = []
        for i in range(top_k):  # for each image in the top k
            if return_entry_list:
                top_k_entries.append(self.knowledge_base[self.faiss_index_ids[I[0][i]]])
            else:
                # find the first knowledge base entry that contains the image
                index_id = self.faiss_index_ids[I[0][i]]
                # return the index of the first element in faiss_index_ids that is equal to index_id
                start_id = self.faiss_index_ids.index(index_id)
                offset = I[0][i] - start_id
                top_k_entries.append(
                    {
                        "url": self.knowledge_base[self.faiss_index_ids[I[0][i]]].url,
                        "knowledge_base_index": self.faiss_index_ids[I[0][i]],
                        "image_url": self.knowledge_base[
                            self.faiss_index_ids[I[0][i]]
                        ].image_urls[offset],
                        "similarity": D[0][i],
                        "kb_entry": self.knowledge_base[self.faiss_index_ids[I[0][i]]],
                    }
                )
        return top_k_entries

    @torch.no_grad()
    def retrieve_image_faiss_batch(self, images, top_k=100, return_entry_list=False):
        """Retrieve the top K similar images from the knowledge base using faiss in batch.

        Args:
            images: The images to be compared.
            top_k (int): The number of top similar images to retrieve.
            return_entry_list (bool): Whether to return the entry list.

        Returns:
            list: Top k entries, every entry is a dict of (url, kb_index, similarity)
        """
        # Process the input image
        if self.model_type == "clip":
            # CLIP Process the input image
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs.to(self.device)
            outputs = self.model(**inputs)
            image_scores = outputs.pooler_output
        elif self.model_type == "eva-clip":
            # EVA-CLIP Process the input image
            inputs = (
                self.processor(images=images, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            image_scores = self.model.encode_image(inputs)
        elif self.model_type in {"clip-mlp", "clip_mlp"}:
            if self.projector is None:
                raise RuntimeError("clip-mlp retriever requires a loaded projector.")
            inputs = (
                self.processor(images=images, return_tensors="pt")
                .pixel_values.to(self.device)
                .half()
            )
            if hasattr(self.model, "get_image_features"):
                student = self.model.get_image_features(pixel_values=inputs)
            else:
                student = self.model(pixel_values=inputs).image_embeds
            student = student / student.norm(dim=-1, keepdim=True)
            image_scores = self.projector(student)
        assert self.faiss_index and self.faiss_index_ids is not None
        query = image_scores.float()
        query = torch.nn.functional.normalize(query, dim=-1)
        query_np = query.detach().cpu().contiguous().numpy().astype("float32")
        Ds, Is = self.faiss_index.search(query_np, top_k)
        top_k_list = []
        for D, I in zip(Ds, Is):
            top_k_entries = []
            for i in range(top_k):
                if return_entry_list:
                    top_k_entries.append(
                        self.knowledge_base[self.faiss_index_ids[I[i]]]
                    )
                else:
                    top_k_entries.append(
                        {
                            "url": self.knowledge_base[self.faiss_index_ids[I[i]]].url,
                            "knowledge_base_index": self.faiss_index_ids[I[i]],
                            "image_urls": self.knowledge_base[
                                self.faiss_index_ids[I[i]]
                            ].image_urls,
                            "similarity": D[i],
                            "kb_entry": self.knowledge_base[self.faiss_index_ids[I[i]]],
                        }
                    )
            top_k_list.append(top_k_entries)

        return top_k_list
