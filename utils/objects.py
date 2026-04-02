from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

import io
import base64
import os
from IPython.display import Image, display

from utils.prompts import (
    SUMMARIZE_IMAGE_SYSTEM_PROMPT,
    SUMMARIZE_TABLE_SYSTEM_PROMPT,
    SUMMARIZE_TEXT_SYSTEM_PROMPT,
    SUMMARIZE_CODE_SYSTEM_PROMPT,
)
from utils.models import LLMModel


def _resolve_llm(llm=None, model_name="gemma3:12b", provider="ollama", **kwargs):
    if llm is not None:
        if hasattr(llm, "as_langchain_llm"):
            return llm.as_langchain_llm()
        return llm
    return LLMModel(
        model_name=model_name, provider=provider, **kwargs
    ).as_langchain_llm()


def _sanitize_text_for_llm(value, max_chars=None):
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = str(value)

    # Remove null bytes and invalid surrogate code points that can break JSON payload parsing.
    text = text.replace("\x00", "")
    text = "".join(ch for ch in text if not 0xD800 <= ord(ch) <= 0xDFFF)
    text = text.encode("utf-8", "replace").decode("utf-8")

    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text


def _encode_token_ids(text, tokenizer):
    encoded = tokenizer.tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return encoded["input_ids"]


def get_central_tokens(text, max_tokens, tokenizer):
    tokens = _encode_token_ids(text, tokenizer)
    total_tokens = len(tokens)

    if total_tokens <= max_tokens:
        return text
    else:
        if total_tokens <= tokenizer.max_tokens:
            return text

    # Calculate middle slice
    start_idx = (total_tokens - max_tokens) // 2
    end_idx = start_idx + max_tokens

    return tokenizer.tokenizer.decode(tokens[start_idx:end_idx])


def build_context_from_sequence(
    elements,
    target_sequence_index,
    tokenizer,
    max_context_len=512,
    target_text=None,
):
    if not elements:
        return ""

    max_tokens = (
        max_context_len if max_context_len is not None else tokenizer.max_tokens
    )
    index_item = None
    target_item = None

    for i, item in enumerate(elements):
        if item.get("sequence_index") == target_sequence_index:
            index_item = i
            target_item = item
            break

    if index_item is None:
        return ""

    if target_text is None:
        target_text = target_item.get("context_text") or target_item.get("content", "")

    target_tokens = len(_encode_token_ids(target_text, tokenizer))
    if target_tokens >= max_tokens:
        return get_central_tokens(target_text, max_tokens, tokenizer)

    remaining_budget = max_tokens - target_tokens
    half_remaining = remaining_budget // 2

    previous_context_text = ""
    current_prev_tokens = 0
    p = index_item - 1
    while p >= 0 and current_prev_tokens < half_remaining:
        item_text = elements[p].get("context_text", "")
        item_tokens = _encode_token_ids(item_text, tokenizer)
        if current_prev_tokens + len(item_tokens) > half_remaining:
            break
        previous_context_text = item_text + "\n" + previous_context_text
        current_prev_tokens += len(item_tokens)
        p -= 1

    post_context_text = ""
    current_post_tokens = 0
    n = index_item + 1
    while (
        n < len(elements)
        and (current_post_tokens + current_prev_tokens) < remaining_budget
    ):
        item_text = elements[n].get("context_text", "")
        item_tokens = _encode_token_ids(item_text, tokenizer)
        if current_post_tokens + len(item_tokens) > (
            remaining_budget - current_prev_tokens
        ):
            break
        post_context_text = post_context_text + "\n" + item_text
        current_post_tokens += len(item_tokens)
        n += 1

    return f"{previous_context_text}\n{target_text}\n{post_context_text}".strip()


class DocumentObject:
    def __init__(
        self,
        document_type,
        tokenizer,
        description=None,
        max_context_len=512,
    ):
        self.document_type = document_type
        self.description = description
        self.tokenizer = tokenizer
        self.max_context_len = max_context_len
        self.metadata = {}

    def find_context_for_element(
        self,
        element_ref,
        converted_document,
    ):
        items = list(converted_document.document.iterate_items())
        # Configuration
        if self.max_context_len is None:
            max_tokens = (
                self.tokenizer.max_tokens
            )  # Assuming your tokenizer wrapper has this attr
        else:
            max_tokens = self.max_context_len

        index_item = None
        target_item = None
        for i, (item, _) in enumerate(items):
            if getattr(item, "self_ref", None) == element_ref:
                index_item = i
                target_item = item
                break

        if index_item is None:
            return ""

        # 1. Start with the target element itself
        # Tables are converted to Markdown automatically by Docling's .text or export
        if self.document_type == "table":
            target_text = target_item.export_to_markdown(
                doc=converted_document.document
            )
        elif self.document_type == "image":
            target_text = getattr(target_item, "text", "")
        target_tokens = len(_encode_token_ids(target_text, self.tokenizer))

        # If the target is already too big, center-crop it and return
        if target_tokens >= max_tokens:
            return get_central_tokens(target_text, max_tokens, self.tokenizer)

        # 2. Distribute remaining budget to context
        remaining_budget = max_tokens - target_tokens
        half_remaining = remaining_budget // 2

        # 3. Enrich Previous Context (Lookback)
        previous_context_text = ""
        current_prev_tokens = 0
        p = index_item - 1
        while p >= 0 and current_prev_tokens < half_remaining:
            item_text = getattr(items[p][0], "text", "")
            item_tokens = _encode_token_ids(item_text, self.tokenizer)

            if current_prev_tokens + len(item_tokens) > half_remaining:
                break

            previous_context_text = item_text + "\n" + previous_context_text
            current_prev_tokens += len(item_tokens)
            p -= 1

        # 4. Enrich Post Context (Lookahead)
        post_context_text = ""
        current_post_tokens = 0
        n = index_item + 1
        while (
            n < len(items)
            and (current_post_tokens + current_prev_tokens) < remaining_budget
        ):
            item_text = getattr(items[n][0], "text", "")
            item_tokens = _encode_token_ids(item_text, self.tokenizer)

            if current_post_tokens + len(item_tokens) > (
                remaining_budget - current_prev_tokens
            ):
                break

            post_context_text = post_context_text + "\n" + item_text
            current_post_tokens += len(item_tokens)
            n += 1

        # Combine: [Prev] + [Target] + [Post]
        return f"{previous_context_text}\n{target_text}\n{post_context_text}".strip()


class TableObject(DocumentObject):
    def __init__(self, element, converted_document, tokenizer):
        self.document_type = "table"
        super().__init__(document_type=self.document_type, tokenizer=tokenizer)
        self.context = self.find_context_for_element(
            element.self_ref, converted_document
        )
        self.markdown = element.export_to_markdown(doc=converted_document.document)
        self.metadata = self.extract_table_item_metadata(
            element, converted_document.document.origin.filename
        )

    def extract_table_item_metadata(self, table_item, doc_filename="unknown"):
        """
        Extracts filename, pages, and bboxes from a Docling TableItem.
        """
        # 1. Pages: Extract unique page numbers from the provenance
        pages = list(set(p.page_no for p in table_item.prov))

        # 2. Bboxes: Extract coordinates for the whole table
        # Typically TableItem has one main bbox in table_item.prov
        bboxes = []
        for prov in table_item.prov:
            b = prov.bbox
            bboxes.append(
                {
                    "page": prov.page_no,
                    "l": b.l,
                    "t": b.t,
                    "r": b.r,
                    "b": b.b,
                    "coord_origin": str(b.coord_origin),
                }
            )

        return {"filename": doc_filename, "pages": sorted(pages), "bboxes": bboxes}

    def summarize_table(
        self,
        system_instruction=SUMMARIZE_TABLE_SYSTEM_PROMPT,
        model_name="gemma3:12b",
        provider="ollama",
        llm=None,
        **llm_kwargs,
    ):
        # We build the message list manually to handle the image block correctly
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Background Context: {self.context}"),
            HumanMessage(content=f"{self.markdown}"),
        ]
        model = _resolve_llm(
            llm=llm,
            model_name=model_name,
            provider=provider,
            **llm_kwargs,
        )
        chain = model | StrOutputParser()
        self.description = chain.invoke(messages)


class ImageObject(DocumentObject):
    def __init__(self, element, converted_document, tokenizer):
        self.document_type = "image"
        super().__init__(document_type=self.document_type, tokenizer=tokenizer)
        self.context = self.find_context_for_element(
            element.self_ref, converted_document
        )
        buffered = io.BytesIO()
        element.image.pil_image.save(buffered, format="PNG")
        base64_result = base64.b64encode(buffered.getvalue()).decode("utf-8")
        self.base64 = base64_result
        self.metadata = self.extract_picture_metadata(
            element, converted_document.document.origin.filename
        )

    def extract_picture_metadata(self, picture_item, doc_filename="unknown"):
        """
        Extracts filename, pages, and bboxes from a Docling PictureItem.
        """
        # 1. Pages: Get unique page numbers where the picture appears
        pages = list(set(p.page_no for p in picture_item.prov))

        # 2. Bboxes: Extract the boundary of the image
        bboxes = []
        for prov in picture_item.prov:
            b = prov.bbox
            bboxes.append(
                {
                    "page": prov.page_no,
                    "l": b.l,
                    "t": b.t,
                    "r": b.r,
                    "b": b.b,
                    "coord_origin": str(b.coord_origin),
                }
            )

        # 3. Image Specifics (Optional but helpful)
        # This detects the format (png) and size of the extracted snippet
        img_ref = getattr(picture_item, "image", None)

        return {
            "filename": doc_filename,
            "pages": sorted(pages),
            "bboxes": bboxes,
            "image_meta": {
                "mimetype": img_ref.mimetype if img_ref else None,
                "width": img_ref.size.width if img_ref else None,
                "height": img_ref.size.height if img_ref else None,
            },
        }

    def display_image(self):
        image_data = base64.b64decode(self.base64)
        display(Image(data=image_data))

    def summarize_image(
        self,
        system_prompt=SUMMARIZE_IMAGE_SYSTEM_PROMPT,
        model_name="gemma3:12b",
        provider="ollama",
        llm=None,
        **llm_kwargs,
    ):
        # We build the message list manually to handle the image block correctly
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Background Context: {self.context}"),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self.base64}"},
                    },
                ]
            ),
        ]
        model = _resolve_llm(
            llm=llm,
            model_name=model_name,
            provider=provider,
            **llm_kwargs,
        )
        chain = model | StrOutputParser()
        self.description = chain.invoke(messages)


class TextChunk:
    def __init__(self, text, metadata):
        self.text = text
        self.description = None
        self.metadata = self.extract_docling_metadata(metadata)

    def extract_docling_metadata(self, meta):
        """
        Extracts high-level and spatial metadata from a Docling DocMeta object.
        """
        if isinstance(meta, dict):
            res = dict(meta)
            if "filename" not in res:
                res["filename"] = "unknown"
            if "pages" not in res:
                res["pages"] = []
            if "bboxes" not in res:
                res["bboxes"] = []
            return res

        # 1. Basic File Info
        origin = getattr(meta, "origin", None)
        res = {
            "filename": getattr(origin, "filename", "unknown"),
            "pages": set(),
            "bboxes": [],
        }

        # 2. Iterate through document items (text blocks, tables, etc.)
        for item in getattr(meta, "doc_items", []):
            for prov in getattr(item, "prov", []):
                # Track which pages this chunk spans
                if hasattr(prov, "page_no"):
                    res["pages"].add(prov.page_no)

                # Extract Bounding Box coordinates
                if hasattr(prov, "bbox"):
                    b = prov.bbox
                    res["bboxes"].append(
                        {
                            "page": prov.page_no,
                            "l": b.l,
                            "t": b.t,
                            "r": b.r,
                            "b": b.b,
                            "coord_origin": str(b.coord_origin),
                        }
                    )

        # Clean up pages to a sorted list
        res["pages"] = sorted(list(res["pages"]))
        return res

    def summarize_text(
        self,
        system_prompt=SUMMARIZE_TEXT_SYSTEM_PROMPT,
        model_name="gemma3:latest",
        provider="ollama",
        llm=None,
        **llm_kwargs,
    ):
        # We build the message list manually to handle the image block correctly
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{self.text}"),
        ]
        model = _resolve_llm(
            llm=llm,
            model_name=model_name,
            provider=provider,
            **llm_kwargs,
        )
        chain = model | StrOutputParser()
        self.description = chain.invoke(messages)


class NotebookTableObject(DocumentObject):
    def __init__(self, markdown, metadata, context, tokenizer):
        self.document_type = "table"
        super().__init__(document_type=self.document_type, tokenizer=tokenizer)
        self.markdown = markdown
        self.metadata = metadata
        self.context = context

    def summarize_table(
        self,
        system_instruction=SUMMARIZE_TABLE_SYSTEM_PROMPT,
        model_name="gemma3:12b",
        provider="ollama",
        llm=None,
        **llm_kwargs,
    ):
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Background Context: {self.context}"),
            HumanMessage(content=f"{self.markdown}"),
        ]
        model = _resolve_llm(
            llm=llm,
            model_name=model_name,
            provider=provider,
            **llm_kwargs,
        )
        chain = model | StrOutputParser()
        self.description = chain.invoke(messages)


class NotebookImageObject(DocumentObject):
    def __init__(self, base64_data, metadata, context, tokenizer):
        self.document_type = "image"
        super().__init__(document_type=self.document_type, tokenizer=tokenizer)
        self.base64 = base64_data
        self.metadata = metadata
        self.context = context

    def summarize_image(
        self,
        system_prompt=SUMMARIZE_IMAGE_SYSTEM_PROMPT,
        model_name="gemma3:12b",
        provider="ollama",
        llm=None,
        **llm_kwargs,
    ):
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Background Context: {self.context}"),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self.base64}"},
                    },
                ]
            ),
        ]
        model = _resolve_llm(
            llm=llm,
            model_name=model_name,
            provider=provider,
            **llm_kwargs,
        )
        chain = model | StrOutputParser()
        self.description = chain.invoke(messages)


class CodeObject:
    content: str
    description: str
    metadata: dict
    context: str
    type: str

    def __init__(self, content, description, metadata, context, type):
        self.content = content
        self.description = description
        self.metadata = metadata
        self.context = context
        self.type = type


class PyDocsObject(CodeObject):
    def __init__(
        self,
        document,
    ):

        content = document.page_content
        description = None
        metadata = document.metadata
        context = None
        document_type = metadata.get("content_type", "python")

        super().__init__(content, description, metadata, context, document_type)

    def summarize_code(
        self,
        system_prompt=SUMMARIZE_CODE_SYSTEM_PROMPT,
        model_name="gemma3:latest",
        provider="ollama",
        llm=None,
        **llm_kwargs,
    ):
        max_input_chars = int(os.getenv("SUMMARY_CODE_MAX_INPUT_CHARS", "12000"))
        safe_code = _sanitize_text_for_llm(self.content, max_chars=max_input_chars)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=safe_code),
        ]
        model = _resolve_llm(
            llm=llm,
            model_name=model_name,
            provider=provider,
            **llm_kwargs,
        )
        chain = model | StrOutputParser()
        self.description = chain.invoke(messages)


class NotebookCodeObject(CodeObject):
    def __init__(self, content, metadata, context):
        super().__init__(
            content=content,
            description=None,
            metadata=metadata,
            context=context,
            type="notebook_code",
        )

    def summarize_code(
        self,
        system_prompt=SUMMARIZE_CODE_SYSTEM_PROMPT,
        model_name="gemma3:latest",
        provider="ollama",
        llm=None,
        **llm_kwargs,
    ):
        max_input_chars = int(os.getenv("SUMMARY_CODE_MAX_INPUT_CHARS", "12000"))
        safe_code = _sanitize_text_for_llm(self.content, max_chars=max_input_chars)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=safe_code),
        ]
        model = _resolve_llm(
            llm=llm,
            model_name=model_name,
            provider=provider,
            **llm_kwargs,
        )
        chain = model | StrOutputParser()
        self.description = chain.invoke(messages)
