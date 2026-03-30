import base64
import csv
import io
import logging
import re
from html import unescape

import imagehash
import nbformat
from PIL import Image as PILImage
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc.labels import DocItemLabel
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from utils.objects import (
    ImageObject,
    NotebookImageObject,
    NotebookTableObject,
    PyDocsObject,
    TableObject,
    TextChunk,
    build_context_from_sequence,
)

LOGGER = logging.getLogger(__name__)

IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
}


def filter_images(base64_list, hash_threshold=10):
    unique_images = []
    unique_hashes = []

    for b64 in base64_list:
        try:
            img_data = base64.b64decode(b64)
            img = (
                PILImage.open(io.BytesIO(img_data))
                .convert("L")
                .resize((32, 32), PILImage.Resampling.LANCZOS)
            )
            current_hash = imagehash.phash(img)
        except Exception:
            if b64 not in unique_images:
                unique_images.append(b64)
            continue

        is_duplicate = False
        for saved_hash in unique_hashes:
            if (current_hash - saved_hash) <= hash_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_images.append(b64)
            unique_hashes.append(current_hash)

    return unique_images


def deduplicate_image_objects(image_objects):
    if len(image_objects) <= 1:
        return image_objects

    unique_base64 = set(filter_images([obj.base64 for obj in image_objects]))
    deduplicated = []
    seen = set()

    for obj in image_objects:
        if obj.base64 in unique_base64 and obj.base64 not in seen:
            deduplicated.append(obj)
            seen.add(obj.base64)

    return deduplicated


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


def _load_tokenizer(tokenizer_model_path):
    LOGGER.info("Loading tokenizer...")
    tok_ = AutoTokenizer.from_pretrained(
        tokenizer_model_path,
        local_files_only=True,
    )
    return HuggingFaceTokenizer(
        tokenizer=tok_,
        max_tokens=tok_.model_max_length,
    )


def _stringify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def _extract_markdown_tables(text):
    lines = text.splitlines()
    tables = []
    i = 0
    separator_pattern = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

    while i < len(lines) - 1:
        line = lines[i]
        next_line = lines[i + 1]

        if "|" in line and separator_pattern.match(next_line):
            block = [line, next_line]
            i += 2
            while i < len(lines):
                candidate = lines[i]
                if "|" not in candidate or candidate.strip() == "":
                    break
                block.append(candidate)
                i += 1
            tables.append("\n".join(block).strip())
        else:
            i += 1

    return tables


def _strip_html_tags(value):
    return unescape(re.sub(r"<[^>]+>", "", value)).strip()


def _extract_html_tables_as_markdown(html_text):
    if not html_text:
        return []

    tables = []
    for table_match in re.findall(r"<table[\\s\\S]*?</table>", html_text, flags=re.IGNORECASE):
        rows = []
        for tr in re.findall(r"<tr[\\s\\S]*?</tr>", table_match, flags=re.IGNORECASE):
            cells = re.findall(r"<t[hd][^>]*>([\\s\\S]*?)</t[hd]>", tr, flags=re.IGNORECASE)
            cleaned = [_strip_html_tags(cell) for cell in cells]
            if cleaned:
                rows.append(cleaned)

        if len(rows) >= 2:
            header = rows[0]
            body = rows[1:]
            header_line = "| " + " | ".join(header) + " |"
            separator = "| " + " | ".join(["---"] * len(header)) + " |"
            body_lines = ["| " + " | ".join(row + [""] * (len(header) - len(row))) + " |" for row in body]
            tables.append("\n".join([header_line, separator] + body_lines))

    return tables


def _csv_to_markdown(csv_text):
    csv_text = csv_text.strip()
    if not csv_text:
        return None

    try:
        rows = list(csv.reader(io.StringIO(csv_text)))
    except Exception:
        return None

    if len(rows) < 2:
        return None

    header = rows[0]
    body = rows[1:]
    header_line = "| " + " | ".join(header) + " |"
    separator = "| " + " | ".join(["---"] * len(header)) + " |"
    body_lines = ["| " + " | ".join(row + [""] * (len(header) - len(row))) + " |" for row in body]
    return "\n".join([header_line, separator] + body_lines)


def _json_tabular_to_markdown(data_obj):
    if isinstance(data_obj, list) and data_obj and all(isinstance(x, dict) for x in data_obj):
        headers = []
        seen = set()
        for row in data_obj:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    headers.append(str(key))

        header_line = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = []
        for row in data_obj:
            values = [str(row.get(h, "")) for h in headers]
            body_lines.append("| " + " | ".join(values) + " |")
        return "\n".join([header_line, separator] + body_lines)

    if isinstance(data_obj, dict) and data_obj:
        keys = list(data_obj.keys())
        columns = [v for v in data_obj.values() if isinstance(v, list)]
        if columns and all(len(v) == len(columns[0]) for v in columns):
            header_line = "| " + " | ".join(str(k) for k in keys) + " |"
            separator = "| " + " | ".join(["---"] * len(keys)) + " |"
            body_lines = []
            for row_idx in range(len(columns[0])):
                row_vals = []
                for key in keys:
                    value = data_obj[key]
                    if isinstance(value, list):
                        row_vals.append(str(value[row_idx]))
                    else:
                        row_vals.append(str(value))
                body_lines.append("| " + " | ".join(row_vals) + " |")
            return "\n".join([header_line, separator] + body_lines)

    return None


def _build_notebook_metadata(path, cell_index, cell_type, sequence_index, output_index=None, mime_type=None):
    return {
        "filename": path,
        "origin": "ipynb",
        "pages": [],
        "bboxes": [],
        "cell_index": cell_index,
        "cell_type": cell_type,
        "output_index": output_index,
        "mime_type": mime_type,
        "sequence_index": sequence_index,
    }


def _normalize_base64_image(data):
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "".join(str(x) for x in data)
    return ""


def ingest_document(
    document_path: str,
    tokenizer_model_path: str = "local_tokenizer/embeddinggemma",
):

    if document_path.endswith(".pdf"):
        return ingest_pdf_document(document_path, tokenizer_model_path)

    if document_path.endswith(".ipynb"):
        return ingest_ipynb_document(document_path, tokenizer_model_path)

    if document_path.endswith(".py"):
        return ingest_py_document(document_path)

    raise ValueError(f"Unsupported file format: {document_path}")


def ingest_pdf_document(
    document_path: str,
    tokenizer_model_path: str = "local_tokenizer/embeddinggemma",
):
    LOGGER.info(
        "Defining the Converter and converting the pdf document with docling..."
    )
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 2.0
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(document_path)
    LOGGER.info("Document converted successfully!")

    tokenizer = _load_tokenizer(tokenizer_model_path)

    LOGGER.info("Creating Hybrid Chunker...")

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=tokenizer.max_tokens / 4,
        serializer_provider=MDTableSerializerProvider(),
    )

    LOGGER.info("Chunking the document")
    chunk_iter = chunker.chunk(dl_doc=result.document)
    chunks = list(chunk_iter)

    LOGGER.info("Parsing the tables and Images")

    table_objs = []
    images_objs = []

    for element, _ in result.document.iterate_items():
        label = element.label
        if label == DocItemLabel.TABLE:
            table_objs.append(
                TableObject(element, converted_document=result, tokenizer=tokenizer)
            )

        if label == DocItemLabel.PICTURE:
            images_objs.append(
                ImageObject(element, converted_document=result, tokenizer=tokenizer)
            )

    LOGGER.info("Removing duplicate images.")
    images_objs = deduplicate_image_objects(images_objs)

    LOGGER.info("Adding text objects")
    text_objs = [TextChunk(chunk.text, chunk.meta) for chunk in chunks]

    return text_objs, table_objs, images_objs


def ingest_ipynb_document(
    document_path: str,
    tokenizer_model_path: str = "local_tokenizer/embeddinggemma",
):
    LOGGER.info("Loading notebook from %s", document_path)
    notebook = nbformat.read(document_path, as_version=4)
    tokenizer = _load_tokenizer(tokenizer_model_path)

    sequence = []
    text_elements = []
    table_candidates = []
    image_candidates = []
    sequence_index = 0

    def add_sequence(kind, content, metadata, context_text=None):
        nonlocal sequence_index
        item = {
            "kind": kind,
            "content": content,
            "context_text": context_text if context_text is not None else content,
            "metadata": metadata,
            "sequence_index": sequence_index,
        }
        sequence.append(item)
        sequence_index += 1
        return item

    for cell_index, cell in enumerate(notebook.cells):
        cell_type = cell.get("cell_type", "unknown")
        source = _stringify(cell.get("source", ""))

        if source.strip():
            content = f"[{cell_type} cell {cell_index}]\n{source}"
            metadata = _build_notebook_metadata(
                document_path,
                cell_index=cell_index,
                cell_type=cell_type,
                sequence_index=sequence_index,
            )
            text_item = add_sequence("text", content, metadata)
            text_elements.append(text_item)

        if cell_type == "markdown":
            for markdown_table in _extract_markdown_tables(source):
                metadata = _build_notebook_metadata(
                    document_path,
                    cell_index=cell_index,
                    cell_type=cell_type,
                    sequence_index=sequence_index,
                    mime_type="text/markdown",
                )
                table_item = add_sequence("table", markdown_table, metadata)
                table_candidates.append(table_item)

            for attachment_name, attachment_data in cell.get("attachments", {}).items():
                if not isinstance(attachment_data, dict):
                    continue
                for mime_type, payload in attachment_data.items():
                    if mime_type not in IMAGE_MIME_TYPES:
                        continue
                    image_b64 = _normalize_base64_image(payload)
                    if not image_b64:
                        continue
                    metadata = _build_notebook_metadata(
                        document_path,
                        cell_index=cell_index,
                        cell_type=cell_type,
                        sequence_index=sequence_index,
                        mime_type=mime_type,
                    )
                    image_item = add_sequence(
                        "image",
                        image_b64,
                        metadata,
                        context_text=f"Markdown attachment '{attachment_name}' in cell {cell_index}",
                    )
                    image_candidates.append(image_item)

        if cell_type != "code":
            continue

        for output_index, output in enumerate(cell.get("outputs", [])):
            output_type = output.get("output_type", "")

            if output_type == "stream":
                stream_text = _stringify(output.get("text", "")).strip()
                if stream_text:
                    metadata = _build_notebook_metadata(
                        document_path,
                        cell_index=cell_index,
                        cell_type=cell_type,
                        sequence_index=sequence_index,
                        output_index=output_index,
                        mime_type="text/plain",
                    )
                    text_item = add_sequence(
                        "text", f"[output stream]\n{stream_text}", metadata
                    )
                    text_elements.append(text_item)

            traceback = output.get("traceback")
            if traceback:
                tb_text = _stringify(traceback).strip()
                metadata = _build_notebook_metadata(
                    document_path,
                    cell_index=cell_index,
                    cell_type=cell_type,
                    sequence_index=sequence_index,
                    output_index=output_index,
                    mime_type="text/plain",
                )
                text_item = add_sequence("text", f"[traceback]\n{tb_text}", metadata)
                text_elements.append(text_item)

            data = output.get("data", {})
            if not isinstance(data, dict):
                continue

            text_plain = _stringify(data.get("text/plain", "")).strip()
            if text_plain:
                metadata = _build_notebook_metadata(
                    document_path,
                    cell_index=cell_index,
                    cell_type=cell_type,
                    sequence_index=sequence_index,
                    output_index=output_index,
                    mime_type="text/plain",
                )
                text_item = add_sequence("text", f"[output text]\n{text_plain}", metadata)
                text_elements.append(text_item)

            markdown_output = _stringify(data.get("text/markdown", "")).strip()
            if markdown_output:
                extracted = _extract_markdown_tables(markdown_output)
                if extracted:
                    for markdown_table in extracted:
                        metadata = _build_notebook_metadata(
                            document_path,
                            cell_index=cell_index,
                            cell_type=cell_type,
                            sequence_index=sequence_index,
                            output_index=output_index,
                            mime_type="text/markdown",
                        )
                        table_item = add_sequence("table", markdown_table, metadata)
                        table_candidates.append(table_item)
                else:
                    metadata = _build_notebook_metadata(
                        document_path,
                        cell_index=cell_index,
                        cell_type=cell_type,
                        sequence_index=sequence_index,
                        output_index=output_index,
                        mime_type="text/markdown",
                    )
                    text_item = add_sequence("text", markdown_output, metadata)
                    text_elements.append(text_item)

            csv_output = _stringify(data.get("text/csv", "")).strip()
            csv_markdown = _csv_to_markdown(csv_output)
            if csv_markdown:
                metadata = _build_notebook_metadata(
                    document_path,
                    cell_index=cell_index,
                    cell_type=cell_type,
                    sequence_index=sequence_index,
                    output_index=output_index,
                    mime_type="text/csv",
                )
                table_item = add_sequence("table", csv_markdown, metadata)
                table_candidates.append(table_item)

            html_output = _stringify(data.get("text/html", "")).strip()
            html_tables = _extract_html_tables_as_markdown(html_output)
            for html_table in html_tables:
                metadata = _build_notebook_metadata(
                    document_path,
                    cell_index=cell_index,
                    cell_type=cell_type,
                    sequence_index=sequence_index,
                    output_index=output_index,
                    mime_type="text/html",
                )
                table_item = add_sequence("table", html_table, metadata)
                table_candidates.append(table_item)

            json_payload = data.get("application/json")
            if json_payload:
                markdown = _json_tabular_to_markdown(json_payload)
                if markdown:
                    metadata = _build_notebook_metadata(
                        document_path,
                        cell_index=cell_index,
                        cell_type=cell_type,
                        sequence_index=sequence_index,
                        output_index=output_index,
                        mime_type="application/json",
                    )
                    table_item = add_sequence("table", markdown, metadata)
                    table_candidates.append(table_item)

            for mime_type, payload in data.items():
                if mime_type not in IMAGE_MIME_TYPES:
                    continue
                image_b64 = _normalize_base64_image(payload)
                if not image_b64:
                    continue

                metadata = _build_notebook_metadata(
                    document_path,
                    cell_index=cell_index,
                    cell_type=cell_type,
                    sequence_index=sequence_index,
                    output_index=output_index,
                    mime_type=mime_type,
                )
                image_item = add_sequence(
                    "image",
                    image_b64,
                    metadata,
                    context_text=f"Output image from code cell {cell_index}. Code context: {source[:500]}",
                )
                image_candidates.append(image_item)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    text_objs = []
    for text_element in text_elements:
        chunks = splitter.split_text(text_element["content"])
        for chunk_idx, chunk_text in enumerate(chunks):
            metadata = dict(text_element["metadata"])
            metadata["chunk_index"] = chunk_idx
            text_objs.append(TextChunk(chunk_text, metadata))

    table_objs = []
    for table_element in table_candidates:
        context = build_context_from_sequence(
            sequence,
            table_element["sequence_index"],
            tokenizer,
            target_text=table_element["content"],
        )
        table_objs.append(
            NotebookTableObject(
                markdown=table_element["content"],
                metadata=table_element["metadata"],
                context=context,
                tokenizer=tokenizer,
            )
        )

    image_objs = []
    for image_element in image_candidates:
        context = build_context_from_sequence(
            sequence,
            image_element["sequence_index"],
            tokenizer,
            target_text=image_element["context_text"],
        )
        image_objs.append(
            NotebookImageObject(
                base64_data=image_element["content"],
                metadata=image_element["metadata"],
                context=context,
                tokenizer=tokenizer,
            )
        )

    image_objs = deduplicate_image_objects(image_objs)

    LOGGER.info(
        "Notebook parsed: %s text chunks, %s tables, %s images",
        len(text_objs),
        len(table_objs),
        len(image_objs),
    )

    return text_objs, table_objs, image_objs


def ingest_py_document(
    document_path: str,
):
    loader = GenericLoader.from_filesystem(
        document_path,
        parser=LanguageParser(language="python"),
    )
    python_docs = loader.load()
    return [PyDocsObject(doc) for doc in python_docs]
