import yaml
import os
from pathlib import Path
from typing import Any, Dict, Tuple
import torch

from .qformer_reranker import qformer_rerank
from .bge_text_reranker import BgeHFTextReranker
from .cross_encoder_reranker import CrossEncoderTextReranker  


SUPPORTED_TYPES = {"qformer", "bge_text", "cross_encoder"}


def load_reranker_config(config_path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Reranker config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def build_reranker(cfg: Dict[str, Any]) -> Tuple[str, Any]:
    r_type = cfg.get("type", "qformer")
    if r_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported reranker type: {r_type}")

    if r_type == "qformer":
        params = {
            "qformer_device": cfg.get("qformer_device", "cuda:0"),
            "qformer_batch": int(cfg.get("qformer_batch", 32)),
            "alpha_1": float(cfg.get("alpha_1", 0.5)),
            "alpha_2": float(cfg.get("alpha_2", 0.5)),
        }
        print(
            "[RerankerCfg]"
            f" type=qformer"
            f" qformer_device={params['qformer_device']}"
            f" qformer_batch={params['qformer_batch']}"
            f" alpha_1={params['alpha_1']}"
            f" alpha_2={params['alpha_2']}"
            f" CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}"
        )
        # qformer uses the functional helper; caller passes other runtime args (question, image, etc.)
        return r_type, params

    if r_type == "bge_text":
        dtype_str = cfg.get("torch_dtype")
        dtype = getattr(torch, dtype_str, torch.float16) if isinstance(dtype_str, str) else torch.float16
        batch_size = int(cfg.get("batch_size", 16))
        reranker = BgeHFTextReranker(
            model_name=cfg.get("model_name", "BAAI/bge-reranker-large"),
            device=cfg.get("bge_device", "cuda:0"),
            torch_dtype=dtype,
            batch_size=batch_size,
        )
        params = {
            "truncate_len": cfg.get("truncate_len", 512),
            "alpha_1": float(cfg.get("alpha_1", 0.5)),
            "alpha_2": float(cfg.get("alpha_2", 0.5)),
        }
        print(
            "[RerankerCfg]"
            f" type=bge_text"
            f" model_name={cfg.get('model_name', 'BAAI/bge-reranker-large')}"
            f" device={cfg.get('bge_device', 'cuda:0')}"
            f" torch_dtype={dtype}"
            f" batch_size={batch_size}"
            f" truncate_len={params['truncate_len']}"
            f" alpha_1={params['alpha_1']}"
            f" alpha_2={params['alpha_2']}"
            f" CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}"
        )
        return r_type, (reranker, params)

    if r_type == "cross_encoder":
        dtype_str = cfg.get("torch_dtype")
        dtype = getattr(torch, dtype_str, torch.float16) if isinstance(dtype_str, str) else torch.float16
        batch_size = int(cfg.get("batch_size", 16))
        reranker = CrossEncoderTextReranker(
            model_name=cfg.get("model_name", "cross-encoder/ms-marco-electra-base"),
            device=cfg.get("ce_device", cfg.get("bge_device", "cuda:0")),
            torch_dtype=dtype,
            batch_size=batch_size,
        )
        params = {
            "truncate_len": cfg.get("truncate_len", 512),
            "alpha_1": float(cfg.get("alpha_1", 0.5)),
            "alpha_2": float(cfg.get("alpha_2", 0.5)),
        }
        print(
            "[RerankerCfg]"
            f" type=cross_encoder"
            f" model_name={cfg.get('model_name', 'cross-encoder/ms-marco-electra-base')}"
            f" device={cfg.get('ce_device', cfg.get('bge_device', 'cuda:0'))}"
            f" torch_dtype={dtype}"
            f" batch_size={batch_size}"
            f" truncate_len={params['truncate_len']}"
            f" alpha_1={params['alpha_1']}"
            f" alpha_2={params['alpha_2']}"
            f" CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '')}"
        )
        return r_type, (reranker, params)

    raise ValueError(f"Unsupported reranker type: {r_type}")
