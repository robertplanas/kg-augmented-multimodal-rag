from calendar import c

from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc.labels import DocItemLabel


from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import NotebookLoader
from transformers import AutoTokenizer

import imagehash
from PIL import Image as PILImage
import io
import os
import base64
import nbformat

from utils.objects import TableObject, ImageObject, TextChunk, PyDocsObject

import logging


from langchain_core.schema import Document
from langchain_core.text_splitter import RecursiveCharacterTextSplitter

LOGGER = logging.getLogger(__name__)


class NotebookImageAwareSplitter:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, notebook_json):
        docs = []

        for i, cell in enumerate(notebook_json["cells"]):
            cell_type = cell.get("cell_type")
            source = "".join(cell.get("source", []))

            # ---- TEXT (markdown + code) ----
            if source.strip():
                base_doc = Document(
                    page_content=source,
                    metadata={
                        "cell_index": i,
                        "cell_type": cell_type,
                    },
                )

                docs.extend(self.text_splitter.split_documents([base_doc]))

            # ---- OUTPUTS ----
            if cell_type == "code":
                for output in cell.get("outputs", []):
                    # TEXT OUTPUTS
                    if "text" in output:
                        text_output = "".join(output["text"])
                        docs.append(
                            Document(
                                page_content=text_output,
                                metadata={"cell_index": i, "type": "output_text"},
                            )
                        )

                    # IMAGE OUTPUTS
                    if "data" in output:
                        for mime, data in output["data"].items():
                            if mime.startswith("image/"):
                                image_data = (
                                    data if isinstance(data, str) else "".join(data)
                                )

                                docs.append(
                                    Document(
                                        page_content="Image output from notebook cell",
                                        metadata={
                                            "type": "image",
                                            "mime_type": mime,
                                            "image_base64": image_data,
                                            "cell_index": i,
                                            "source_code": source[
                                                :500
                                            ],  # optional context
                                        },
                                    )
                                )

        return docs


def filter_images(base64_list, hash_threshold=10):
    unique_images = []
    unique_hashes = []

    for b64 in base64_list:
        # Decode and hash
        img_data = base64.b64decode(b64)

        # Use the ALIAS here to avoid the AttributeError
        img = (
            PILImage.open(io.BytesIO(img_data))
            .convert("L")
            .resize((32, 32), PILImage.Resampling.LANCZOS)
        )

        current_hash = imagehash.phash(img)

        is_duplicate = False
        for saved_hash in unique_hashes:
            if (current_hash - saved_hash) <= hash_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_images.append(b64)
            unique_hashes.append(current_hash)

    return unique_images


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # configuring a different table serializer
        )


def ingest_document(
    document_path: str,
    tokenizer_model_path: str = "local_tokenizer/embeddinggemma",
):

    if document_path.endswith(".pdf"):
        return ingest_pdf_document(document_path, tokenizer_model_path)

    if document_path.endswith(".ipynb"):
        return ingest_ipynb_document(document_path, tokenizer_model_path)

    if document_path.endswith(".py"):
        return ingest_ipynb_document(document_path, tokenizer_model_path)

    else:
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

    LOGGER.info("Loading tokenizer...")
    tok_ = AutoTokenizer.from_pretrained(
        tokenizer_model_path,
        # Optional: ensure it doesn't try to download anything if path is missing
        local_files_only=True,
    )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=tok_,
        max_tokens=tok_.model_max_length,
    )

    LOGGER.info("Creating Hybrid Chunker...")

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=tok_.model_max_length / 4,
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

    ### Filter out duplicate items

    LOGGER.info("Removing duplicate images.")
    base64_list = [x.base64 for x in images_objs]
    unique_base_64 = filter_images(base64_list)
    images_objs = [x for x in images_objs if x.base64 in unique_base_64]

    LOGGER.info("Adding text objects")
    text_objs = [TextChunk(chunk.text, chunk.meta) for chunk in chunks]

    return text_objs, table_objs, images_objs


def ingest_ipynb_document(
    document_path: str,
    tokenizer_model_path: str = "local_tokenizer/embeddinggemma",
):

    # Now, use LangChain's loader on the CLEANED notebook
    loader = NotebookLoader(
        document_path, include_outputs=True, max_output_length=9999999
    )
    result = loader.load()
    assert len(result) == 1
    result = result[0]

    LOGGER.info("Loading tokenizer...")
    tok_ = AutoTokenizer.from_pretrained(
        tokenizer_model_path,
        # Optional: ensure it doesn't try to download anything if path is missing
        local_files_only=True,
    )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=tok_,
        max_tokens=tok_.model_max_length,
    )

    LOGGER.info("Creating Hybrid Chunker...")

    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=tok_.model_max_length / 4,
        serializer_provider=MDTableSerializerProvider(),
    )

    LOGGER.info("Chunking the document")
    chunk_iter = chunker.chunk(dl_doc=result)
    chunks = list(chunk_iter)

    return chunks


def ingest_py_document(
    document_path: str,
):
    loader = GenericLoader.from_filesystem(
        document_path,
        parser=LanguageParser(language="python"),
    )
    python_docs = loader.load()
    return [PyDocsObject(doc) for doc in python_docs]
