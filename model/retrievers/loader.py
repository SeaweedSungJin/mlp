import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml

from ..retriever import ClipRetriever


SUPPORTED_TYPES = {"eva_clip", "eva-clip", "clip_mlp", "clip-mlp"}


def _project_root() -> Path:
    # .../SCOOP-RAG/model/retrievers/loader.py -> .../SCOOP-RAG
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    # Prefer CWD for interactive runs, fallback to project root for script runs.
    cand = (Path.cwd() / p).resolve()
    if cand.exists():
        return cand
    return (base / p).resolve()


def load_retriever_config(config_path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Retriever config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_retriever(
    cfg: Dict[str, Any],
    *,
    fallback_type: str = "eva-clip",
    fallback_device: str = "cuda:0",
    fallback_faiss_gpu_id: int = 3,
) -> Tuple[str, ClipRetriever, Dict[str, Any]]:
    """Build a retriever from YAML config.

    Returns: (retriever_type, retriever, params)
    where params includes faiss_gpu_id.
    """
    cfg_type = (cfg.get("type") or fallback_type) if cfg is not None else fallback_type
    cfg_type = str(cfg_type).strip().lower()
    if cfg_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported retriever type: {cfg_type} (supported: {sorted(SUPPORTED_TYPES)})")

    device = str(cfg.get("device", fallback_device))
    dtype_str = cfg.get("torch_dtype", "float16")
    torch_dtype = getattr(torch, dtype_str, torch.float16) if isinstance(dtype_str, str) else torch.float16
    if not torch.cuda.is_available() and torch_dtype in (torch.float16, torch.bfloat16):
        torch_dtype = torch.float32
    faiss_gpu_id = int(cfg.get("faiss_gpu_id", fallback_faiss_gpu_id))

    project_root = _project_root()

    if cfg_type in {"eva_clip", "eva-clip"}:
        model_name = cfg.get("model_name", "BAAI/EVA-CLIP-8B")
        retriever = ClipRetriever(
            model="eva-clip",
            device=device,
            torch_dtype=torch_dtype,
            model_name=model_name,
        )
        r_type = "eva_clip"
        print(
            "[RetrieverCfg]"
            f" type=eva_clip"
            f" model_name={model_name}"
            f" device={device}"
            f" torch_dtype={torch_dtype}"
            f" faiss_gpu_id={faiss_gpu_id}"
            f" CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}"
        )
        return r_type, retriever, {"faiss_gpu_id": faiss_gpu_id}

    if cfg_type in {"clip_mlp", "clip-mlp"}:
        student_model_name = cfg.get("student_model_name", "openai/clip-vit-large-patch14-336")
        projector_path_raw = cfg.get("projector_path", "./eva_projector.pth")
        projector_path = _resolve_path(projector_path_raw, base=project_root)
        if not projector_path.exists():
            raise FileNotFoundError(f"Projector checkpoint not found: {projector_path}")
        retriever = ClipRetriever(
            model="clip-mlp",
            device=device,
            torch_dtype=torch_dtype,
            student_model_name=student_model_name,
            projector_path=str(projector_path),
        )
        r_type = "clip_mlp"
        print(
            "[RetrieverCfg]"
            f" type=clip_mlp"
            f" student_model_name={student_model_name}"
            f" projector_path={projector_path}"
            f" device={device}"
            f" torch_dtype={torch_dtype}"
            f" faiss_gpu_id={faiss_gpu_id}"
            f" CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}"
        )
        return r_type, retriever, {"faiss_gpu_id": faiss_gpu_id, "projector_path": str(projector_path)}

    raise ValueError(f"Unsupported retriever type: {cfg_type}")

