from argparse import ArgumentParser
import json
import os
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional, Sequence, Tuple

import torch
import PIL
from PIL import Image
from torchvision import transforms
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from config.runtime_config import RuntimeConfig
from model import (
    ClipRetriever,
    reconstruct_wiki_article,
    reconstruct_wiki_sections,
    WikipediaKnowledgeBaseEntry,
    BgeTextReranker,
)
from model.retrievers.loader import load_retriever_config, build_retriever
from model.rerankers.loader import load_reranker_config, build_reranker
from model.rerankers.qformer_reranker import qformer_rerank
from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates
from utils.sample_logger import SampleLogger
from nli import NLIConfig, NLISelector, SectionCandidate
from nli.utils import parse_section_payload
from router import Router, RouterConfig

iNat_image_path = "/dataset/inaturalist"


def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20, 100]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def _deduplicate_with_scores(texts: Sequence[str], scores: Sequence[float]) -> Tuple[List[str], List[float]]:
    seen = set()
    out_t: List[str] = []
    out_s: List[float] = []
    for t, s in zip(texts, scores):
        if t in seen:
            continue
        seen.add(t)
        out_t.append(t)
        out_s.append(float(s))
    return out_t, out_s

def _move_inputs_to_device(inputs: dict, device: torch.device):
    out = {}
    for k, v in inputs.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def run_test(
    test_file_path: str,
    knowledge_base_path: str,
    faiss_index_path: str,
    top_ks: list,
    retrieval_top_k: int,
    *,
    dataset_start: int = 0,
    dataset_end: Optional[int] = None,
    dataset_limit: Optional[int] = None,
    nli_selector: Optional[NLISelector] = None,
    nli_section_limit: int = 10,
    nli_context_sentences: int = 3,
    router: Optional[Router] = None,
    **kwargs,
):
    # === Dataset slicing ===
    test_list, test_header = load_csv_data(test_file_path)
    total_examples = len(test_list)
    start_raw = 0 if dataset_start is None else int(dataset_start)
    end_raw = dataset_end
    limit_raw = dataset_limit
    start_idx = max(0, start_raw)
    end_idx = total_examples if end_raw is None else min(total_examples, int(end_raw))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    if limit_raw is not None:
        if limit_raw <= 0:
            print("dataset_limit <= 0, nothing to run.")
            return
        end_idx = min(end_idx, start_idx + int(limit_raw))
    selected_indices = list(range(start_idx, end_idx))
    if not selected_indices:
        print("No examples selected for evaluation.")
        return

    with open(iNat_image_path + "/val_id2name.json", "r") as f:
        iNat_id2name = json.load(f)

    runtime_cfg: RuntimeConfig = kwargs.get("runtime_config") or RuntimeConfig.default()
    sample_logger = SampleLogger(
        runtime_cfg.samples_dir,
        prefix=runtime_cfg.samples_prefix,
        pretty=runtime_cfg.samples_pretty_json,
    ) if runtime_cfg.log_samples else None
    router_meta = kwargs.get("router_meta") or {}
    project_root = Path(__file__).resolve().parent.parent

    # === Retriever or resume ===
    if kwargs["resume_from"] is not None:
        resumed_results = json.load(open(kwargs["resume_from"], "r"))
        kb_dict = json.load(open(knowledge_base_path, "r"))
    else:
        retriever_device = kwargs.get("retriever_device", "cuda:0")
        retriever_cfg_path = Path(kwargs.get("retriever_cfg", "config/retriever_config.yaml"))
        if not retriever_cfg_path.is_absolute():
            retriever_cfg_path = (project_root / retriever_cfg_path).resolve()
        retriever_cfg = load_retriever_config(retriever_cfg_path) if retriever_cfg_path.exists() else {}
        retriever_type, retriever, retriever_params = build_retriever(
            retriever_cfg,
            fallback_type=kwargs.get("retriever_vit", "eva-clip"),
            fallback_device=retriever_device,
        )
        print(f"[Retriever] type={retriever_type} cfg={retriever_cfg_path}")
        retriever.load_knowledge_base(knowledge_base_path)
        retriever.load_faiss_index(faiss_index_path, int(retriever_params.get("faiss_gpu_id", 3)))

    recalls = {k: 0 for k in top_ks}
    reranked_recalls = {k: 0 for k in top_ks}
    hits = 0
    eval_score = 0
    vqa_total_count = 0
    vqa_correct_count = 0
    question_generator = None

    evaluate_answers = bool(kwargs.get("perform_vqa"))
    need_generation = evaluate_answers or runtime_cfg.log_samples

    # === Evaluation ===
    evaluate_example_fn = None
    if evaluate_answers:
        from utils import evaluate_example
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        evaluate_example_fn = evaluate_example

    # === Answer generator ===
    if need_generation:
        # ---- 장치 고정 ----
        qwen_device_str = 'cuda:0'
        qwen_device = torch.device(qwen_device_str if torch.cuda.is_available() else "cpu")
        # retriever_device = torch.device(retriever_device_str if torch.cuda.is_available() else "cpu")
        # qformer_device = torch.device(qformer_device_str if torch.cuda.is_available() else "cpu")

        # ---- qwen 로드 (cuda:0) ----
        print(f"[LLaVA] Loading Qwen/Qwen2.5-VL-7B-Instruct on {qwen_device}")
        qwen_model_id = kwargs.get("model_path","Qwen/Qwen2.5-VL-7B-Instruct")
        qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_id, torch_dtype="auto", device_map={"": qwen_device_str}
        )
        qwen_processor = AutoProcessor.from_pretrained(qwen_model_id)
        print("[LLaVA] Model ready.")

    # === Helper functions ===
    def _generate_answer_with_image(
        qwen_model,
        qwen_processor,
        qwen_device: torch.device,
        question: str,
        image_path: str,
        context_text: str = "",
        *,
        max_new_tokens: int | None = None,
    ):
        """Generate answer using Qwen2.5-VL (image + context + question)."""

        max_tokens = int(max_new_tokens or 64)
        answer = None

        try:
            # === Load image ===
            raw_image = Image.open(image_path).convert("RGB")

            # === Build user text ===
            if context_text.strip():
                user_text = (
                    "You are a knowledgeable visual reasoning assistant.\n"
                    "Analyze the given image together with the textual context provided below.\n"
                    "Use the context as the primary factual source.\n"
                    "If conflicting information appears, rely mainly on the FIRST context item.\n"
                    f"---\nContext:\n{context_text.strip()}\n---\n\n"
                    f"Question:\n{question.strip()}\n\n"
                    "Answer concisely based only on the context and image."
                )
            else:
                user_text = (
                    "You are a visual reasoning assistant.\n"
                    "Look carefully at the image and answer the question concisely.\n\n"
                    f"Question:\n{question.strip()}\n\n"
                    "Answer directly."
                )

            # === Qwen message format ===
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]

            # === Prepare text prompt ===
            prompt_text = qwen_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # === Prepare vision inputs (Qwen-specific) ===
            image_inputs, video_inputs = process_vision_info(messages)

            # === Processor (token + vision) ===
            inputs = qwen_processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to(qwen_device)

            # === Generation ===
            with torch.inference_mode():
                generated_ids = qwen_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.0,
                )

            # === Remove input tokens (Qwen official post-processing) ===
            generated_ids_trimmed = [
                gen[len(inp):]
                for inp, gen in zip(inputs.input_ids, generated_ids)
            ]

            decoded = qwen_processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            answer = decoded[0].strip()

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Qwen2.5] generation failed: {e}")
            answer = None

        return answer




    def _log_sample(sample_idx: int, *, answer: Optional[str], target_answer: List[str],
                    use_rag: bool, context_sections: List[str], reranked_sections: List[str],
                    top_urls: List[str], router_prob: Optional[float], router_backend: Optional[str],
                    image_path: str, question: str) -> None:
        if not sample_logger:
            return
        payload = {
            "question": question,
            "image_path": image_path,
            "router_prob": router_prob,
            "router_threshold": router_meta.get("threshold"),
            "router_backend": router_backend,
            "use_rag": bool(use_rag),
            "answer": answer,
            "target_answer": target_answer,
            "context_sections": context_sections[:runtime_cfg.samples_max_sections],
            "retrieval": {
                "use_retrieval": bool(use_rag),
                "top_urls": top_urls[:runtime_cfg.samples_max_sections],
                "sections": reranked_sections[:runtime_cfg.samples_max_sections],
            },
        }
        sample_logger.log(sample_idx, payload)

    # === Optional rerankers ===
    reranker_cfg_path = Path(kwargs.get("reranker_cfg", "config/reranker_config.yaml"))
    if not reranker_cfg_path.is_absolute():
        reranker_cfg_path = (project_root / reranker_cfg_path).resolve()
    reranker_cfg = load_reranker_config(reranker_cfg_path)
    reranker_type, reranker_obj = build_reranker(reranker_cfg)
    print(f"[Reranker] type={reranker_type} cfg={reranker_cfg_path}")

    text_reranker = None
    if kwargs["perform_text_rerank"]:
        text_reranker = BgeTextReranker(
            model_path="/remote-home/share/huggingface_model/bge-reranker-v2-m3",
            device="cuda:0",
        )

    blip_model = None
    preprocess = None
    txt_processors = None
    qformer_device = kwargs.get("qformer_device", "cuda:0")

    if reranker_type == "qformer" and kwargs["perform_qformer_reranker"]:
        from lavis.models import load_model_and_preprocess
        from data_utils import targetpad_transform
        preprocess = targetpad_transform(1.25, 224)
        blip_model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_reranker",
            model_type="pretrain",
            is_eval=True,
            device="meta",
        )

        checkpoint_path = kwargs["qformer_ckpt_path"]

        blip_model = blip_model.to_empty(device=qformer_device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        vocab_keys = ["Qformer.cls.predictions.bias", "Qformer.cls.predictions.decoder.weight"]
        for key in vocab_keys:
            if key in checkpoint:
                target_param = blip_model.state_dict().get(key, None)
                if target_param is None:
                    continue
                src_tensor = checkpoint[key]
                tgt_shape = target_param.shape
                if src_tensor.shape != tgt_shape:
                    print(f"[QFormer] Reshaping {key} from {tuple(src_tensor.shape)} to {tuple(tgt_shape)}")
                    if src_tensor.numel() > target_param.numel():
                        checkpoint[key] = src_tensor[: tgt_shape[0]] if src_tensor.dim() == 1 else src_tensor[: tgt_shape[0], : tgt_shape[1]]
                    else:
                        padded = torch.zeros_like(target_param)
                        if src_tensor.dim() == 1:
                            padded[: src_tensor.shape[0]] = src_tensor
                        else:
                            padded[: src_tensor.shape[0], : src_tensor.shape[1]] = src_tensor
                        checkpoint[key] = padded

        msg = blip_model.load_state_dict(checkpoint, strict=False)

        blip_model = blip_model.half()

        blip_model.use_vanilla_qformer = True

    metric = "url matching"
    
    # === Evaluation loop ===
    print(f"[Slice] dataset_start={start_idx}, dataset_end={end_idx}, total={len(selected_indices)}")
    router_true_count = router_false_count = 0
    retrieval_result = {}

    for processed_idx, dataset_idx in tqdm(
        enumerate(selected_indices),
        total=len(selected_indices),
        desc=f"Processing {len(selected_indices)} samples",
        unit="sample",
        dynamic_ncols=True,
    ):
        example = get_test_question(dataset_idx, test_list, test_header)
        image_path = get_image(example["dataset_image_ids"].split("|")[0], example["dataset_name"], iNat_id2name)
        image = PIL.Image.open(image_path)
        ground_truth = example["wikipedia_url"]
        target_answer = example["answer"].split("|")
        data_id = example["data_id"] if example["dataset_name"] == "infoseek" else f"E-VQA_{dataset_idx}"
        count_so_far = processed_idx + 1

        # === Router decision ===
        use_rag, router_prob, router_backend = True, None, None
        if router is not None:
            try:
                decision = router.score(example["question"], image_path)
                router_prob = float(decision.prob)
                router_backend = decision.backend
                use_rag = bool(decision.use_rag)
            except Exception as exc:
                print(f"[Router] failed ({exc}); default use_rag=True")

        if not use_rag:
            router_false_count += 1
        else:
            router_true_count += 1

        # === Retrieval (if use_rag) ===
        top_k_wiki, sections, reranked_sections, retrieval_similarities = [], [], [], []
        if use_rag:
            if kwargs["resume_from"] is not None:
                resumed_result = resumed_results[data_id]
                top_k_wiki = resumed_result.get("retrieved_entries", [])
                reranked_sections = resumed_result.get("reranked_sections", [])
                retrieval_similarities = resumed_result.get("retrieval_similarities", [])
                entries = [WikipediaKnowledgeBaseEntry(kb_dict[url]) for url in top_k_wiki]
            else:
                top_k = retriever.retrieve_image_faiss(image, top_k=retrieval_top_k)
                top_k_wiki = remove_list_duplicates([retrieved_entry["url"] for retrieved_entry in top_k])
                entries = remove_list_duplicates([retrieved_entry["kb_entry"] for retrieved_entry in top_k])
                seen = set()
                retrieval_similarities = [
                    float(top_k[i]["similarity"]) for i in range(retrieval_top_k)
                    if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))
                ]
            count_so_far = processed_idx + 1
            if kwargs["save_result"]:
                retrieval_result[data_id] = {
                    "retrieved_entries": [entry.url for entry in entries[:20]],
                    "retrieval_similarities": [float(sim) for sim in retrieval_similarities[:20]],
                    "router_prob": router_prob,
                    "router_backend": router_backend or ("enabled" if router is not None else "disabled"),
                    "use_rag": use_rag,
                }
            # Build section list
            if kwargs["resume_from"] is None:
                sections = [] ; section_to_entry: List[int] = []
                for entry_id, entry in enumerate(entries):
                    entry_sections = reconstruct_wiki_sections(entry)
                    sections.extend(entry_sections)
                    section_to_entry.extend([entry_id] * len(entry_sections))
            else:
                sections = list(reranked_sections) ; section_to_entry = list(range(len(sections)))
            # Parent scores
            section_parent_scores = [ float(retrieval_similarities[idx]) if idx < len(retrieval_similarities) else 0.0 for idx in section_to_entry ] if sections else []
            reranked_sections = list(sections)
            reranked_scores = list(section_parent_scores)
            # Doc recall (before NLI)
            if metric == "answer matching":
                entry_articles = [reconstruct_wiki_article(entry) for entry in entries]
                found = False
                for i, entry in enumerate(entry_articles):
                    if any(ans.strip().lower() in entry.strip().lower() for ans in target_answer):
                        found = True ; break
                if found:
                    for k in top_ks:
                        if i < k:
                            recalls[k] += 1
            else:
                recall = eval_recall(top_k_wiki, ground_truth, top_ks)
                for k in top_ks:
                    recalls[k] += recall[k]
            # for k in top_ks:
                # print("Avg Recall@{}: ".format(k), recalls[k] / count_so_far)

            # Reranker (Q-Former or BGE text) 선택 실행
            if reranker_type == "qformer" and kwargs["perform_qformer_reranker"]:
                alpha_1 = float(reranker_obj.get("alpha_1", 0.5))
                alpha_2 = float(reranker_obj.get("alpha_2", 0.5))
                batch_size = int(reranker_obj.get("qformer_batch", kwargs.get("qformer_batch", 32)))
                top_k_wiki, reranked_sections, reranked_scores = qformer_rerank(
                    question=example["question"],
                    image=image,
                    sections=sections,
                    entries=entries,
                    section_to_entry=section_to_entry,
                    retrieval_similarities=retrieval_similarities,
                    section_parent_scores=section_parent_scores,
                    preprocess=preprocess,
                    blip_model=blip_model,
                    txt_processors=txt_processors,
                    qformer_device=qformer_device,
                    qformer_batch=batch_size,
                    alpha_1=alpha_1,
                    alpha_2=alpha_2,
                    initial_top_k_wiki=top_k_wiki,
                )
                if kwargs["save_result"]:
                    retrieval_result[data_id]["reranked_sections"] = reranked_sections[:10]
            elif reranker_type in {"bge_text", "cross_encoder"}:
                text_reranker, text_params = reranker_obj
                sections_sorted, scores_sorted, idx_sorted = text_reranker.rerank(
                    example["question"],
                    reranked_sections,
                    truncate_len=text_params.get("truncate_len", 512),
                    return_indices=True,
                )
                retrieval_scores = [
                    retrieval_similarities[section_to_entry[i]] if section_to_entry[i] < len(retrieval_similarities) else 0.0
                    for i in idx_sorted
                ]
                alpha_1 = float(text_params.get("alpha_1", 0.5))
                alpha_2 = float(text_params.get("alpha_2", 0.5))
                combined_scores = [alpha_1 * r + alpha_2 * s for r, s in zip(retrieval_scores, scores_sorted)]
                reranked_sections, reranked_scores = _deduplicate_with_scores(sections_sorted, combined_scores)
                if kwargs["save_result"]:
                    retrieval_result[data_id]["reranked_sections"] = reranked_sections[:10]
            else:
                reranked_sections, reranked_scores = _deduplicate_with_scores(reranked_sections, reranked_scores)

            # Cap sections entering NLI
            sec_cap = max(1, int(nli_section_limit)) if nli_section_limit is not None else None
            if sec_cap is not None and reranked_sections:
                reranked_sections = reranked_sections[:sec_cap]
                reranked_scores = reranked_scores[:sec_cap]

            # NLI
            if nli_selector is not None and reranked_sections:
                candidates: List[SectionCandidate] = []
                for idx, (payload, sc) in enumerate(zip(reranked_sections, reranked_scores)):
                    doc_title, section_title, body = parse_section_payload(payload)
                    candidates.append(SectionCandidate(text=body, doc_title=doc_title or "unknown", section_title=section_title or f"section_{idx}", similarity=float(sc), section_index=idx))
                selected = nli_selector.select(example["question"], candidates)
                if selected:
                    # reranked_sections = [f"# Wiki Article: {s.doc_title}\n## Section Title: {s.section_title} [sent {s.sentence_index}]\n{s.text}" for s in selected]
                    reranked_sections = [f"#{s.doc_title}\n##{s.section_title}\n{s.text}" for s in selected]
                    # reranked_sections = [f"{s.text}" for s in selected]
                    reranked_scores = [s.parent_score for s in selected]
                    # print(f"[NLI] Selected {len(reranked_sections)} sentences from {len(candidates)} candidates")

            # Reranked recall (URL)
            recall = eval_recall(top_k_wiki, ground_truth, top_ks)
            for k in top_ks:
                reranked_recalls[k] += recall[k]
            # for k in top_ks:
                # print("Reranked Avg Recall@{}: ".format(k), reranked_recalls[k] / count_so_far)

            if kwargs["perform_text_rerank"]:
                if ground_truth in top_k_wiki[:5]:
                    gt_index = top_k_wiki.index(ground_truth)
                    index, hit = text_reranker.rerank_entry_sections(example["question"], reranked_sections, top_k=5, gt_index=gt_index)
                    temp = reranked_sections[0] ; reranked_sections[0] = reranked_sections[index] ; reranked_sections[index] = temp
                else:
                    hit = 0
                hits += hit ; print("Text Reranking Recalls", hits / count_so_far)

            if kwargs["save_result"]:
                retrieval_result[data_id]["reranked_sections"] = reranked_sections[:10]

        # === Context for generation ===
        if use_rag and reranked_sections:
            ctx_n = max(1, int(nli_context_sentences))
            ctx_sections = reranked_sections[:ctx_n]
        else:
            ctx_sections = []
        ctx_text = "\n\n".join(ctx_sections)

        # === Answer generation ===
        answer = None
        if need_generation:
            answer = _generate_answer_with_image(
                qwen_model,
                qwen_processor,
                qwen_device,
                example["question"],
                image_path,
                ctx_text,
                max_new_tokens=runtime_cfg.vlm_max_new_tokens
            )
            print(f">>> answer: {answer}")
            print(f">>> GT ans: {target_answer}")
            
            if evaluate_answers and answer and evaluate_example_fn is not None:
                score = evaluate_example_fn(
                    example["question"], reference_list=target_answer, candidate=answer,
                    question_type=example["question_type"],
                )
                eval_score += score
                vqa_total_count += 1
                if score >= 0.5:
                    vqa_correct_count += 1
                print(f"score={score:.3f}, eval_avg={eval_score/count_so_far:.3f}")

            if sample_logger and (runtime_cfg.log_samples or evaluate_answers):
                _log_sample(
                    dataset_idx, answer=answer, target_answer=target_answer,
                    use_rag=use_rag, context_sections=ctx_sections,
                    reranked_sections=reranked_sections if use_rag else [],
                    top_urls=top_k_wiki, router_prob=router_prob,
                    router_backend=router_backend, image_path=image_path,
                    question=example["question"],
                )

        if kwargs["save_result"]:
            retrieval_result[data_id]["answer"] = answer

    # === Save & Summary ===
    if kwargs["save_result"]:
        with open(kwargs["save_result_path"], "w") as f:
            json.dump(retrieval_result, f, indent=4)

    if evaluate_answers and vqa_total_count > 0:
        avg_bem = eval_score / vqa_total_count
        acc = vqa_correct_count / vqa_total_count
        print("========== Final VQA Summary ==========")
        print(f"Total: {vqa_total_count}, Correct: {vqa_correct_count}, Acc: {acc*100:.2f}%, Avg BEM: {avg_bem:.4f}")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--top_ks", type=str, default="1,5,10,20,100")
    parser.add_argument("--perform_vqa", action="store_true")
    parser.add_argument("--answer_generator", type=str, default="mistral")
    parser.add_argument("--llm_checkpoint", type=str, default=None)
    parser.add_argument("--retriever_device", type=str, default="cuda:0")
    parser.add_argument("--qformer_device", type=str, default="cuda:0")
    parser.add_argument("--qformer_batch", type=int, default=32)
    parser.add_argument("--llm_device", type=str, default="cuda:0")
    parser.add_argument("--llm_device_map", type=str, default="single")
    parser.add_argument("--llm_dtype", type=str, default="float16")
    parser.add_argument("--llm_load_in_8bit", action="store_true")
    parser.add_argument("--llm_load_in_4bit", action="store_true")
    parser.add_argument("--llm_max_memory", type=str, default=None)
    parser.add_argument("--perform_text_rerank", action="store_true")
    parser.add_argument("--perform_qformer_reranker", action="store_true")
    parser.add_argument("--qformer_ckpt_path", type=str, default=None)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--save_result_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--retriever_vit", type=str, default="clip")
    # New slicing + NLI controls
    parser.add_argument("--dataset_start", type=int, default=0)
    parser.add_argument("--dataset_end", type=int, default=None)
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--enable_router", action="store_true")
    parser.add_argument("--router_config", type=str, default=None)
    parser.add_argument("--router_threshold", type=float, default=None)
    parser.add_argument("--router_backend", type=str, default=None)
    parser.add_argument("--disable_router", action="store_true")
    parser.add_argument("--runtime_config", type=str, default=None)
    parser.add_argument("--enable_nli", action="store_true")
    parser.add_argument("--nli_config", type=str, default=None)
    parser.add_argument("--nli_model", type=str, default=None)
    parser.add_argument("--nli_device", type=str, default=None)
    parser.add_argument("--nli_question_threshold", type=float, default=None)
    parser.add_argument("--nli_tau", type=float, default=None)
    parser.add_argument("--nli_target_size", type=int, default=None)
    parser.add_argument("--nli_section_limit", type=int, default=20)
    parser.add_argument("--nli_context_sentences", type=int, default=5)
    args = parser.parse_args()

    # Env fallbacks for slicing and enabling NLI
    def _env_int(name: str, default_val):
        v = os.getenv(name)
        if v is None or v == "":
            return default_val
        try:
            return int(v)
        except ValueError:
            return default_val
    def _env_bool(name: str, default_val: bool) -> bool:
        v = os.getenv(name)
        if v is None:
            return default_val
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    # Encourage CUDA segmented allocator to reduce fragmentation on large models
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    nli_selector = None
    dataset_start = _env_int("DATASET_START", args.dataset_start)
    dataset_end = _env_int("DATASET_END", args.dataset_end)
    dataset_limit = _env_int("DATASET_LIMIT", args.dataset_limit)
    enable_nli = args.enable_nli or _env_bool("ENABLE_NLI", False)

    # ------------------------------------------------------------------
    # Multi-GPU defaults for large LLMs (3090 x 4)
    # If user didn't explicitly request a mapping, prefer sharding across GPUs.
    # Also distribute Q-Former and retriever to reduce contention.
    # ------------------------------------------------------------------
    try:
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        gpu_count = 0

    # Effective LLM device map: default to 'auto' when multiple GPUs are available
    llm_device_map_req = os.getenv("LLM_DEVICE_MAP", None) or args.llm_device_map
    if (llm_device_map_req is None or llm_device_map_req.strip().lower() == "single") and gpu_count >= 2:
        llm_device_map_eff = "auto"
    else:
        llm_device_map_eff = llm_device_map_req

    # Provide a default max_memory map when sharding
    llm_max_memory_eff = args.llm_max_memory
    if (llm_max_memory_eff is None or llm_max_memory_eff == "") and llm_device_map_eff == "auto" and gpu_count >= 2:
        # Conservative per-GPU budget for 24GiB cards (3090) to avoid OOM
        # e.g., "0:20GiB,1:20GiB,2:20GiB,3:20GiB,cpu:120GiB"
        parts = [f"{i}:20GiB" for i in range(gpu_count)] + ["cpu:120GiB"]
        llm_max_memory_eff = ",".join(parts)

    # Spread smaller models (retriever, Q-Former) to reduce overlap with LLM shards
    retriever_device_eff = args.retriever_device
    qformer_device_eff = args.qformer_device
    if gpu_count >= 2:
        # If defaults are still on cuda:0, move them
        if retriever_device_eff == "cuda:0":
            retriever_device_eff = f"cuda:{min(2, gpu_count-1)}"  # prefer GPU 2 when available
        if qformer_device_eff == "cuda:0":
            qformer_device_eff = f"cuda:{min(1, gpu_count-1)}"  # prefer GPU 1
    # If NLI is enabled but PERFORM_VQA not explicitly set, default to running VQA
    perform_vqa_effective = args.perform_vqa or _env_bool("PERFORM_VQA", False) or enable_nli

    def _env_flag(name: str) -> Optional[bool]:
        v = os.getenv(name)
        if v is None:
            return None
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    router_cfg_path = args.router_config or os.getenv("ROUTER_CONFIG_PATH")
    router_cfg = RouterConfig.from_yaml(router_cfg_path) if router_cfg_path else RouterConfig()

    enable_router = router_cfg.enabled
    env_enable = _env_flag("ENABLE_ROUTER")
    if env_enable is not None:
        enable_router = env_enable
    env_disable = _env_flag("DISABLE_ROUTER")
    if env_disable:
        enable_router = False
    if args.enable_router:
        enable_router = True
    if getattr(args, "disable_router", False):
        enable_router = False

    runtime_cfg_path = args.runtime_config or os.getenv("RUNTIME_CONFIG")
    if runtime_cfg_path:
        runtime_cfg = RuntimeConfig.from_yaml(runtime_cfg_path)
    else:
        runtime_cfg = RuntimeConfig.default()

    router_instance: Optional[Router] = None
    if enable_router:
        backend_override = args.router_backend or os.getenv("ROUTER_BACKEND")
        if backend_override:
            router_cfg.backend = str(backend_override).lower()
        threshold_override = None
        if args.router_threshold is not None:
            threshold_override = args.router_threshold
        else:
            env_threshold = os.getenv("ROUTER_THRESHOLD")
            if env_threshold:
                threshold_override = float(env_threshold)
        if threshold_override is not None:
            router_cfg.threshold = float(threshold_override)
        # For mmengine backend, require 'config_path' to be set inside router YAML.
        # Do not confuse the router YAML path with the mmengine model config path.
        if router_cfg.backend == "mmengine" and router_cfg.config_path is not None:
            # Normalise; if relative, try resolving under project root as well
            cfg_path = Path(str(router_cfg.config_path)).expanduser()
            if not cfg_path.is_absolute():
                project_root = Path(__file__).resolve().parent.parent
                alt_path = (project_root / cfg_path).resolve()
                if alt_path.exists():
                    cfg_path = alt_path
            else:
                cfg_path = cfg_path.resolve()
            if not cfg_path.exists():
                print(f"[Router] mmengine config_path not found: {cfg_path}. Falling back to heuristic backend.")
                router_cfg.backend = "heuristic"
            else:
                router_cfg.config_path = cfg_path
        # Optional env override for router checkpoint
        if os.getenv("ROUTER_CHECKPOINT") and not getattr(router_cfg, "router_checkpoint", None):
            router_cfg.router_checkpoint = Path(os.getenv("ROUTER_CHECKPOINT")).expanduser().resolve()
        if router_cfg.backend == "mmengine" and router_cfg.config_path is None:
            print("[Router] mmengine backend requested but config_path is missing. Falling back to heuristic backend.")
            router_cfg.backend = "heuristic"
        router_instance = Router(router_cfg)
        print("[DEBUG] router backend:", type(router_instance._backend))
        print(f"[Router] enabled backend={router_instance.backend_name} threshold={router_cfg.threshold}")

    if enable_nli:
        nli_cfg = NLIConfig.from_yaml(args.nli_config) if args.nli_config else NLIConfig()
        if args.nli_model:
            nli_cfg.model_name = args.nli_model
        if args.nli_device:
            nli_cfg.device = args.nli_device
        if args.nli_question_threshold is not None:
            nli_cfg.question_entail_threshold = args.nli_question_threshold
        if args.nli_tau is not None:
            nli_cfg.tau = args.nli_tau
        if args.nli_target_size is not None:
            nli_cfg.target_size = args.nli_target_size
        if dataset_start is not None:
            nli_cfg.dataset_start = dataset_start
        if dataset_end is not None:
            nli_cfg.dataset_end = dataset_end
        if dataset_limit is not None:
            nli_cfg.dataset_limit = dataset_limit
        dataset_start = nli_cfg.dataset_start
        dataset_end = nli_cfg.dataset_end
        dataset_limit = getattr(nli_cfg, "dataset_limit", dataset_limit)
        nli_selector = NLISelector(nli_cfg)

    test_config = {
        "test_file_path": args.test_file,
        "knowledge_base_path": args.knowledge_base,
        "faiss_index_path": args.faiss_index,
        "model_path": args.model_path,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "perform_vqa": perform_vqa_effective,
        "answer_generator": args.answer_generator,
        "llm_checkpoint": args.llm_checkpoint,
        "retriever_device": retriever_device_eff,
        "qformer_device": qformer_device_eff,
        "qformer_batch": args.qformer_batch,
        "llm_device": args.llm_device,
        "llm_device_map": llm_device_map_eff,
        "llm_max_memory": llm_max_memory_eff,
        "llm_dtype": args.llm_dtype,
        "llm_load_in_8bit": args.llm_load_in_8bit,
        "llm_load_in_4bit": args.llm_load_in_4bit,
        "perform_text_rerank": args.perform_text_rerank,
        "perform_qformer_reranker": args.perform_qformer_reranker,
        "qformer_ckpt_path": args.qformer_ckpt_path,
        "save_result": args.save_result,
        "save_result_path": args.save_result_path,
        "resume_from": args.resume_from,
        "retriever_vit": args.retriever_vit,
        "dataset_start": dataset_start,
        "dataset_end": dataset_end,
        "dataset_limit": dataset_limit,
        "nli_selector": nli_selector,
        "nli_section_limit": args.nli_section_limit,
        "nli_context_sentences": args.nli_context_sentences,
        "router": router_instance,
        "runtime_config": runtime_cfg,
        "router_meta": {
            "enabled": enable_router,
            "threshold": router_cfg.threshold if enable_router else router_cfg.threshold,
            "backend": router_cfg.backend,
        },
    }
    debug_config = {k: v for k, v in test_config.items() if k != "nli_selector"}
    debug_config["enable_nli"] = enable_nli
    debug_config["perform_vqa"] = perform_vqa_effective
    debug_config["llm_device_map"] = llm_device_map_eff
    debug_config["llm_max_memory"] = llm_max_memory_eff
    debug_config["retriever_device"] = retriever_device_eff
    debug_config["qformer_device"] = qformer_device_eff
    debug_config["enable_router"] = enable_router
    debug_config["router_backend"] = router_instance.backend_name if router_instance is not None else router_cfg.backend if enable_router else None
    debug_config["router_threshold"] = router_cfg.threshold if enable_router else None
    debug_config["router_config_path"] = str(router_cfg.config_path) if router_cfg.config_path else router_cfg_path
    debug_config["runtime_config"] = runtime_cfg.to_dict() if hasattr(runtime_cfg, "to_dict") else {}
    print(f"[Runner] file={__file__}")
    print("test_config: ", debug_config)
    run_test(**test_config)
