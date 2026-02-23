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

from transformers import AutoTokenizer

import imagehash
from PIL import Image as PILImage
import io
import base64

from utils.objects import TableObject, ImageObject, TextChunk

import logging

LOGGER = logging.getLogger(__name__)


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
