""" Serve for question generation for a list of given knowledge base entries. 
"""

from openai import OpenAI
import torch
from .retriever import WikipediaKnowledgeBaseEntry
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.generation import GenerationConfig

import time


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def reconstruct_wiki_article(knowledge_entry: WikipediaKnowledgeBaseEntry):
    """Reconstruct the wiki article from the knowledge entry class."""
    title = knowledge_entry.title
    article = "# Wiki Article: " + title + "\n"
    for it, section_title in enumerate(knowledge_entry.section_titles):
        if (
            "external link" in section_title.lower()
            or "reference" in section_title.lower()
        ):
            continue
        article += (
            "\n## Section Title: "
            + section_title
            + "\n"
            + knowledge_entry.section_texts[it]
        )

    return article


def reconstruct_wiki_sections(knowledge_entry, section_index=-1):
    """Reconstruct the wiki sections from the knowledge entry class."""
    title = knowledge_entry.title
    sections = []

    def skip_section(title, text):
        t = title.lower()
        # 불필요한 섹션 제목 필터링
        if any(
            x in t
            for x in [
                "external links",
                "references",
                "see also",
                "citations",
                "further reading",
                "notes",
                "list",  # ← 추가 (목록형 섹션 제외)
                "gallery",
                "bibliography",
                "timeline"
            ]
        ):
            return True
        # 내용 기반 필터링
        if text.strip().startswith(("Retrieved", "Accessed", "Archived", "ISBN", "DOI")):
            return True
        if len(text.strip()) < 40:  # 너무 짧은 section 제외
            return True
        return False

    for it, section_title in enumerate(knowledge_entry.section_titles):
        section_text = knowledge_entry.section_texts[it]
        if skip_section(section_title, section_text):
            continue

        if it == int(section_index):
            evidence_section = (
                f"# Wiki Article: {title}\n"
                f"## Section Title: {section_title}\n"
                f"{section_text}"
            )
        else:
            sections.append(
                f"# Wiki Article: {title}\n"
                f"## Section Title: {section_title}\n"
                f"{section_text}"
            )

    if section_index != -1:
        return evidence_section, sections
    return sections



def get_all_sections(knowledge_entry):
    """Get all sections in list format."""
    sections = []
    for it, section_title in enumerate(knowledge_entry.section_titles):
        sections.append(
            "* Section Title: "
            + section_title
            + "\n"
            + knowledge_entry.section_texts[it]
        )

    return sections


pseudo_tokenizer = None


def _adjust_prompt_length(prompt, desired_token_length):
    """Adjust the prompt length to the desired token length."""
    global pseudo_tokenizer

    if pseudo_tokenizer is None:
        pseudo_tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2", use_fast=False
        )

    # Tokenize the prompt
    tokens = pseudo_tokenizer.encode(prompt)

    if len(tokens) > desired_token_length:
        # If the prompt is too long, trim it
        trimmed_tokens = tokens[:desired_token_length]
        # Convert tokens back to text
        trimmed_text = pseudo_tokenizer.decode(
            trimmed_tokens, clean_up_tokenization_spaces=True
        )[4:]
        return trimmed_text
    else:
        # If the length is already as desired
        return prompt


def _build_multimodal_prompt(question: str, context: str = "", include_image_hint: bool = False) -> str:
    """Builds the USER content for chat models.

    Context-aware prompt formatted as requested:
    - With context:
        Context: <CONTEXT>
        Question: <QUESTION>
        Just answer the question, no explanations.
        Short answer is:
    - Without context:
        Question: <QUESTION>
        Just answer the question, no explanations.
        Short answer is:
    """
    context = (context or "").strip()
    parts = []
    if context:
        parts.append(f"Context:\n{context}")
    parts.append(f"Question: {question.strip()}")
    parts.append("Just answer the question, no explanations.")
    parts.append("Short answer is:")
    return "\n".join(parts)


class AnswerGenerator:
    """Question generator for EchoSight."""

    def __init__(self):
        self.model = None

    def load_model(self, model_name):
        """Load the model.

        Args:
            model_name: The model to load.
        """
        raise NotImplementedError


class MistralAnswerGenerator(AnswerGenerator):
    """Mistral/Vicuna (HF CausalLM) answer generator for EchoSight."""

    def __init__(
        self,
        device,
        model_path,
        use_embedding_model=False,
        device_map=None,
        load_in_8bit=False,
        load_in_4bit=False,
        max_memory=None,
        torch_dtype: str | None = "bfloat16",
    ):
        """Initialize the generator.

        Args:
            device: single-device string like 'cuda:0'. Ignored if device_map is not None.
            model_path: HF repo id or local path.
            use_embedding_model: deprecated.
            device_map: None for single GPU, or 'auto' to shard across GPUs/CPU.
            load_in_8bit: quantize with 8-bit (bitsandbytes required).
            load_in_4bit: quantize with 4-bit (bitsandbytes required).
            max_memory: optional max memory map, dict or string "0:20GiB,1:20GiB,cpu:60GiB".
            torch_dtype: 'float16' | 'bfloat16' | 'auto'.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        self.torch_dtype = torch_dtype
        self._load_model()
        if use_embedding_model:
            self._load_embedding()
        else:
            self.emb = None

    def _parse_max_memory(self, mm):
        if mm is None:
            return None
        if isinstance(mm, dict):
            return mm
        if isinstance(mm, str):
            out = {}
            for kv in mm.split(","):
                k, v = kv.split(":")
                k = k.strip()
                v = v.strip()
                if k.isdigit():
                    out[int(k)] = v
                else:
                    out[k] = v
            return out
        return None

    def _resolve_dtype(self):
        if self.torch_dtype is None or self.torch_dtype == "auto":
            return None
        td = self.torch_dtype.lower()
        if td in ("fp16", "float16", "half"):
            return torch.float16
        if td in ("bf16", "bfloat16"):
            return torch.bfloat16
        return None

    def _load_model(self):
        """Load the model with optional sharding/quantization."""
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        kwargs = {"low_cpu_mem_usage": True}
        dtype = self._resolve_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        # ------------------------------------------------------------------
        # Multi-GPU friendly defaults
        # If device_map is not provided (single), prefer sharding across GPUs
        # when available. Also provide a conservative max_memory map.
        # ------------------------------------------------------------------
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        eff_device_map = self.device_map
        if (eff_device_map is None or (isinstance(eff_device_map, str) and eff_device_map.strip().lower() == "single")) and gpu_count >= 2:
            eff_device_map = "auto"
        if eff_device_map is not None:
            kwargs["device_map"] = eff_device_map
            mm = self._parse_max_memory(self.max_memory)
            if mm is None and isinstance(eff_device_map, str) and eff_device_map == "auto":
                # Default to ~20GiB per 24GiB card to avoid OOM
                mm = {i: "20GiB" for i in range(gpu_count)}
                mm["cpu"] = "120GiB"
            if mm is not None:
                kwargs["max_memory"] = mm
        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
            # optional 4-bit config
            kwargs["bnb_4bit_compute_dtype"] = dtype or torch.float16
            kwargs["bnb_4bit_quant_type"] = "nf4"
            kwargs["bnb_4bit_use_double_quant"] = True

        # Try loading, with graceful fallbacks on OOM
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs).eval()
        except torch.cuda.OutOfMemoryError:
            # Fallback 1: enable 8-bit if not already set
            if not self.load_in_8bit:
                kwargs["load_in_8bit"] = True
                kwargs.pop("load_in_4bit", None)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs).eval()
                except torch.cuda.OutOfMemoryError:
                    # Fallback 2: force 'auto' map with CPU offload and tighter per-GPU budget
                    kwargs["device_map"] = "auto"
                    mm = {i: "18GiB" for i in range(gpu_count)} if gpu_count > 0 else {}
                    mm["cpu"] = "160GiB"
                    kwargs["max_memory"] = mm
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs).eval()
            else:
                # Already in 8-bit: try with CPU offload + smaller budget
                kwargs["device_map"] = "auto"
                mm = {i: "18GiB" for i in range(gpu_count)} if gpu_count > 0 else {}
                mm["cpu"] = "160GiB"
                kwargs["max_memory"] = mm
                self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs).eval()

        # Only move if not using device_map at all
        if (self.device_map is None or (isinstance(self.device_map, str) and self.device_map.strip().lower() == "single")) \
            and kwargs.get("device_map") is None and hasattr(self, "device") and self.device:
            self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(
        self,
        question,
        entry=None,
        entry_dict=None,
        entry_section=None,
        image_path=None,
        oracle_setting="subject",
        evidence_sec=None,
    ):
        """Answer the question for a given entry

        Args:
            question: The question to answer.
            entry: The entry to answer the question for.
            entry_dict: The entry dictionary to answer the question for.
            entry_section: The entry section to answer the question for.
            oracle_setting: The setting for the oracle experiment.
            evidence_sec: The evidence section.
        """
        context = ""
        if entry is not None:
            context = reconstruct_wiki_article(entry)
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
        elif entry_section is not None:
            context = entry_section
        if context:
            context = _adjust_prompt_length(context, 4096)

        prompt = _build_multimodal_prompt(question, context, include_image_hint=bool(image_path))

        messages = [
            {
                "role": "system",
                "content": "You always answer exactly the asked question. No extra text.",
            },
            {"role": "user", "content": prompt},
        ]

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        encodeds = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", max_length=8000, truncation=True
        )
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = self.tokenizer.decode(
            generated_ids[0][model_inputs.shape[1] :], skip_special_tokens=True
        )

        return response.strip()


class LLaMA3AnswerGenerator(AnswerGenerator):
    def __init__(
        self,
        device,
        model_path,
        device_map=None,
        load_in_8bit=False,
        load_in_4bit=False,
        max_memory=None,
        torch_dtype: str | None = "bfloat16",
    ):
        """Initialize the QuestionGenerator class.

        Args:
            device: The device to use for the model.
            model_path: The model to load.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        self.torch_dtype = torch_dtype
        self._load_model()

    def _parse_max_memory(self, mm):
        if mm is None:
            return None
        if isinstance(mm, dict):
            return mm
        if isinstance(mm, str):
            out = {}
            for kv in mm.split(","):
                k, v = kv.split(":")
                k = k.strip()
                v = v.strip()
                if k.isdigit():
                    out[int(k)] = v
                else:
                    out[k] = v
            return out
        return None

    def _resolve_dtype(self):
        if self.torch_dtype is None or self.torch_dtype == "auto":
            return None
        td = self.torch_dtype.lower()
        if td in ("fp16", "float16", "half"):
            return torch.float16
        if td in ("bf16", "bfloat16"):
            return torch.bfloat16
        return None

    def _load_model(self):
        """Load the model with optional sharding/quantization."""
        disable_torch_init()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        kwargs = {"low_cpu_mem_usage": True}
        dtype = self._resolve_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
            mm = self._parse_max_memory(self.max_memory)
            if mm is not None:
                kwargs["max_memory"] = mm
        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
            kwargs["bnb_4bit_compute_dtype"] = dtype or torch.float16
            kwargs["bnb_4bit_quant_type"] = "nf4"
            kwargs["bnb_4bit_use_double_quant"] = True

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs).eval()
        if self.device_map is None and hasattr(self, "device") and self.device:
            self.model.to(self.device)

    @torch.no_grad()
    def llm_answering(
        self,
        question,
        entry=None,
        entry_dict=None,
        entry_section=None,
        image_path=None,
        oracle_setting="subject",
        evidence_sec=None,
    ):
        """Answer the question for a given entry

        Args:
            question: The question to answer.
            entry: The entry to answer the question for.
            entry_dict: The entry dictionary to answer the question for.
            entry_section: The entry section to answer the question for.
            oracle_setting: The setting for the oracle experiment.
            evidence_sec: The evidence section.
        """
        context = ""
        if entry is not None:
            context = reconstruct_wiki_article(entry)
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
        elif entry_section is not None:
            context = entry_section
        if context:
            context = _adjust_prompt_length(context, 4096)

        prompt = _build_multimodal_prompt(question, context, include_image_hint=bool(image_path))

        messages = [
            {
                "role": "system",
                "content": "You are a concise encyclopedic assistant. Answer with the most relevant fact grounded in the provided context and image. Do not add explanations.",
            },
            {"role": "user", "content": prompt},
        ]
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        encodeds = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", max_length=8000, truncation=True
        )
        model_inputs = encodeds.to(self.device)
        generated_ids = self.model.generate(
            model_inputs,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = self.tokenizer.decode(
            generated_ids[0][model_inputs.shape[1] :], skip_special_tokens=True
        )

        return response.strip()


class GPT4AnswerGenerator(AnswerGenerator):
    """OpenAI Question generator for EchoSight."""

    def __init__(self):
        """Initialize the QuestionGenerator class."""
        super().__init__()
        self.client = OpenAI(api_key="YOUR_API_KEY")

    def get_gpt4_answer(self, input):
        """Get the answer from the GPT-4 model.

        Args:
            input: The input to the model.
        """
        MAX_RETRIES = 5
        retries = 0

        while retries < MAX_RETRIES:
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input},
                    ],
                )

                assistant_reply = completion.choices[0].message.content
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                retries += 1
                time.sleep(2)
        return assistant_reply

    def llm_answering(self, question, entry=None, entry_dict=None, entry_section=None, image_path=None):
        """Answer the question for a given entry

        Args:
            question: The question to answer.
            entry: The entry to answer the question for.
            entry_dict: The entry dictionary to answer the question for.
            entry_section: The entry section to answer the question for.
        """
        context = ""
        if entry is not None:
            context = reconstruct_wiki_article(entry)
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
        elif entry_section is not None:
            context = entry_section
        if context:
            context = _adjust_prompt_length(context, 4096)

        prompt = _build_multimodal_prompt(question, context, include_image_hint=bool(image_path))
        # Embed the desired SYSTEM line by sending it via system role in get_gpt4_answer
        # Adjust get_gpt4_answer to use strict system instruction.
        response = self.get_gpt4_answer(
            f'SYSTEM: "You always answer exactly the asked question. No extra text."\nUSER:\n{prompt}'
        )
        return response


class PaLMAnswerGenerator(AnswerGenerator):
    """Google PaLM Question generator for EchoSight."""

    def __init__(self):
        """Initialize the QuestionGenerator class."""
        super().__init__()
        import vertexai
        from vertexai.preview.language_models import (
            ChatModel,
            InputOutputTextPair,
            TextEmbeddingModel,
            TextGenerationModel,
        )
        import os

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_CREDENTIALS.json"
        PROJECT_ID = "YOUR_PROJECT_ID"
        REGION = "YOUR_REGION"
        vertexai.init(project=PROJECT_ID, location=REGION)
        self.model = TextGenerationModel.from_pretrained("text-bison@002")

    def llm_answering(self, question, entry=None, entry_dict=None, entry_section=None, image_path=None):
        """Answer the question for a given entry

        Args:
            question: The question to answer.
            entry: The entry to answer the question for.
            entry_dict: The entry dictionary to answer the question for.
            entry_section: The entry section to answer the question for.
        """
        context = ""
        if entry is not None:
            context = reconstruct_wiki_article(entry)
        elif entry_dict is not None:
            context = reconstruct_wiki_article(WikipediaKnowledgeBaseEntry(entry_dict))
        elif entry_section is not None:
            context = entry_section
        if context:
            context = _adjust_prompt_length(context, 4096)

        prompt = _build_multimodal_prompt(question, context, include_image_hint=bool(image_path))

        response = self.model.predict(
            f'SYSTEM: "You always answer exactly the asked question. No extra text."\nUSER:\n{prompt}',
            temperature=0.2,
            max_output_tokens=128,
            top_k=40,
            top_p=0.95,
        ).text
        return response


class BgeTextReranker:
    """Text Reranker for EchoSight.

    Deprecated in the current version.
    """

    def __init__(self, model_path, device):
        """Initialize the Text Reranker"""
        self.device = device
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank_entry_sections(self, question, sections, top_k=3, gt_index=-1):
        if gt_index == -1:
            return -1, 0
        pairs = [[question, section] for section in sections[:top_k]]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=6000
        ).to(self.device)
        scores = (
            self.model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
        )
        scores, index = torch.sort(scores, descending=True)

        return index[0], int(index[0]) == int(gt_index)
