import torch
from torch.cuda.amp import autocast
from typing import List, Sequence, Tuple, Optional

from utils import remove_list_duplicates


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


def qformer_rerank(
    *,
    question: str,
    image,
    sections: Sequence[str],
    entries: Sequence,
    section_to_entry: Sequence[int],
    retrieval_similarities: Sequence[float],
    preprocess,
    blip_model,
    txt_processors,
    qformer_device,
    section_parent_scores: Optional[Sequence[float]] = None,
    qformer_batch: int = 32,
    alpha_1: float = 0.5,
    alpha_2: float = 0.5,
    initial_top_k_wiki: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], List[float]]:
    """Run Q-Former-based reranking.

    Returns (top_k_wiki, reranked_sections, reranked_scores).
    In skip cases (no sections/entries/embeddings), falls back to inputs.
    """

    reranked_sections = list(sections)
    reranked_scores = list(section_parent_scores or [])
    top_k_wiki = list(initial_top_k_wiki or [])

    if len(sections) == 0 or len(entries) == 0:
        return top_k_wiki, reranked_sections, reranked_scores

    qformer_articles = [txt_processors["eval"](article) for article in sections]
    if len(qformer_articles) == 0:
        return top_k_wiki, reranked_sections, reranked_scores

    reference_image = preprocess(image).to(qformer_device).unsqueeze(0)

    with autocast():
        fusion_embs = blip_model.extract_features(
            {"image": reference_image, "text_input": question},
            mode="multimodal",
        )["multimodal_embeds"]

        rerank_step = int(qformer_batch)
        article_embs_all = None

        for sp in range(0, len(qformer_articles), rerank_step):
            batch = qformer_articles[sp : sp + rerank_step]
            if len(batch) == 0:
                continue
            embs = blip_model.extract_features({"text_input": batch}, mode="text")[
                "text_embeds_proj"
            ][:, 0, :]
            article_embs_all = embs if article_embs_all is None else torch.cat(
                (article_embs_all, embs), dim=0
            )

        if article_embs_all is None or article_embs_all.size(0) == 0:
            return top_k_wiki, reranked_sections, reranked_scores

        scores = torch.matmul(
            article_embs_all.unsqueeze(1).unsqueeze(1), fusion_embs.permute(0, 2, 1)
        ).squeeze()
        scores, _ = scores.max(-1)

        section_similarities = [
            retrieval_similarities[section_to_entry[i]]
            if section_to_entry[i] < len(retrieval_similarities)
            else 0.0
            for i in range(len(sections))
        ]

        scores = (
            alpha_1 * torch.tensor(section_similarities, device=qformer_device)
            + alpha_2 * scores
        )
        scores, reranked_index = torch.sort(scores, descending=True)

    top_k_wiki = remove_list_duplicates(
        [entries[section_to_entry[i]].url for i in reranked_index]
    )
    ranked_sections = [sections[i] for i in reranked_index]
    ranked_scores = scores.cpu().tolist()

    reranked_sections, reranked_scores = _deduplicate_with_scores(
        ranked_sections, ranked_scores
    )

    return top_k_wiki, reranked_sections, reranked_scores
