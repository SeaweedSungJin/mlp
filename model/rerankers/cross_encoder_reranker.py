import torch
from torch.cuda.amp import autocast
from typing import List, Sequence, Tuple, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderTextReranker:
    """Text-only reranker using HF cross-encoder models (e.g., ms-marco electra)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-electra-base",
        device: str = "cuda:0",
        torch_dtype: Union[torch.dtype, str, None] = torch.float16,
        batch_size: int = 16,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype, torch.float16)
        if self.device.type != "cuda" and torch_dtype in (torch.float16, torch.bfloat16):
            torch_dtype = torch.float32
        self.torch_dtype = torch_dtype
        self.use_autocast = self.device.type == "cuda" and self.torch_dtype in (torch.float16, torch.bfloat16)
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def rerank(
        self,
        question: str,
        sections: Sequence[str],
        *,
        truncate_len: int = 512,
        return_indices: bool = False,
    ) -> Tuple[List[str], List[float]] | Tuple[List[str], List[float], List[int]]:
        """Return sections sorted by relevance score to the question."""
        if not sections:
            return ([], [], []) if return_indices else ([], [])

        scores_chunks: List[torch.Tensor] = []
        for start in range(0, len(sections), self.batch_size):
            batch_sections = sections[start : start + self.batch_size]
            pairs_a = [question for _ in batch_sections]
            pairs_b = list(batch_sections)
            inputs = self.tokenizer(
                pairs_a,
                pairs_b,
                padding=True,
                truncation=True,
                max_length=truncate_len,
                return_tensors="pt",
            ).to(self.device)
            if self.use_autocast:
                with autocast(dtype=self.torch_dtype):
                    logits = self.model(**inputs, return_dict=True).logits
            else:
                logits = self.model(**inputs, return_dict=True).logits
            scores_chunks.append(logits.view(-1).float().cpu())

        scores = torch.cat(scores_chunks, dim=0)
        sorted_idx = torch.argsort(scores, descending=True)

        sections_sorted = [sections[int(i)] for i in sorted_idx]
        scores_sorted = [float(scores[int(i)].item()) for i in sorted_idx]
        indices_sorted = [int(i) for i in sorted_idx]

        if return_indices:
            return sections_sorted, scores_sorted, indices_sorted
        return sections_sorted, scores_sorted

    @torch.no_grad()
    def rerank_entry_sections(
        self,
        question: str,
        sections: Sequence[str],
        top_k: int = 3,
        gt_index: int = -1,
    ) -> Tuple[int, int]:
        ranked, _, _ = self.rerank(question, sections[:top_k], return_indices=True)
        if not ranked:
            return -1, 0
        best_section = ranked[0]
        try:
            idx = sections.index(best_section)
        except ValueError:
            idx = -1
        hit = int(idx == gt_index) if gt_index != -1 else 0
        return idx, hit
