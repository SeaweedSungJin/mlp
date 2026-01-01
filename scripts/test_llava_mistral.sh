#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default to all 4 GPUs unless user explicitly sets it
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# 1) 모델 선택
MODEL_PATH="llava-hf/llava-v1.6-mistral-7b-hf"

# 2) 데이터셋 선택 (evqa / infoseek)
DATASET_NAME="evqa"

# 데이터셋 루트 경로
DATA_ROOT="/dataset"

if [ "$DATASET_NAME" == "evqa" ]; then
  BASE_PATH="${DATA_ROOT}/evqa"
  KB_JSON="${BASE_PATH}/encyclopedic_kb_wiki.json"
  FAISS_INDEX_DIR="${BASE_PATH}/"
  TEST_FILE="${BASE_PATH}/test.csv"
elif [ "$DATASET_NAME" == "infoseek" ]; then
  BASE_PATH="${DATA_ROOT}/infoseek"
  KB_JSON="${BASE_PATH}/wiki_100_dict_v4.json"
  FAISS_INDEX_DIR="${BASE_PATH}/"
  TEST_FILE="${BASE_PATH}/infoseek_test_new_filtered.csv"
else
  echo "Error: Unknown DATASET_NAME: $DATASET_NAME"
  exit 1
fi

QFORMER_CKPT="${QFORMER_CKPT:-${PROJECT_ROOT}/reranker.pth}"
DEFAULT_RUNTIME_CONFIG="${PROJECT_ROOT}/config/runtime.yaml"
if [[ ! -f "${DEFAULT_RUNTIME_CONFIG}" ]]; then
  DEFAULT_RUNTIME_CONFIG="${PROJECT_ROOT}/config/runtime.example.yaml"
fi
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${DEFAULT_RUNTIME_CONFIG}}"
DATASET_START="${DATASET_START:-}"
DATASET_END="${DATASET_END:-}"
DATASET_LIMIT="${DATASET_LIMIT:-}"
RETRIEVER_VIT="${RETRIEVER_VIT:-eva-clip}"
PERFORM_VQA="${PERFORM_VQA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  python3 "${PROJECT_ROOT}/test/test_llava_mistral.py"
  --test_file "${TEST_FILE}"
  --knowledge_base "${KB_JSON}"
  --faiss_index "${FAISS_INDEX_DIR}"
  --model_path "${MODEL_PATH}"
  --retriever_vit "${RETRIEVER_VIT}"
  --top_ks 1,5,10,20
  --retrieval_top_k 20
  --perform_qformer_reranker
  --qformer_ckpt_path "${QFORMER_CKPT}"
  --perform_vqa
)

if [[ -n "${RUNTIME_CONFIG}" && -f "${RUNTIME_CONFIG}" ]]; then
  CMD+=(--runtime_config "${RUNTIME_CONFIG}")
fi

if [[ -n "${DATASET_START}" ]]; then
  CMD+=(--dataset_start "${DATASET_START}")
fi
if [[ -n "${DATASET_END}" ]]; then
  CMD+=(--dataset_end "${DATASET_END}")
fi
if [[ -n "${DATASET_LIMIT}" ]]; then
  CMD+=(--dataset_limit "${DATASET_LIMIT}")
fi
if [[ -n "${PERFORM_VQA}" ]]; then
  CMD+=(--perform_vqa)
fi
# Allow arbitrary passthrough flags via EXTRA_ARGS
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  CMD+=( ${EXTRA_ARGS} )
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" "${CMD[@]}"
