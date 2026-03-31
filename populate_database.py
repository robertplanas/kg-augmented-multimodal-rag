import argparse
import glob
import logging
from pathlib import Path
from typing import List, Tuple

from utils.database_utils import generate_database_and_retriever, populate_database
from utils.ingest import ingest_document
from utils.summarize import summarize_objects


LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS: Tuple[str, ...] = ("pdf", "ipynb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate DB")
    parser.add_argument(
        "-df",
        "--document_folder",
        type=str,
        help="Folder where the documents that need parsing are stored.",
        required=True,
    )
    parser.add_argument(
        "-db",
        "--data_base",
        type=str,
        default="./localdb",
        help="Folder where the DB will be stored.",
    )
    return parser.parse_args()


def get_documents(folder: str) -> List[str]:
    documents: List[str] = []
    for extension in SUPPORTED_EXTENSIONS:
        documents.extend(glob.glob(f"{folder}/*.{extension}"))
    return sorted(documents)


def process_documents(documents: List[str]):
    all_texts = []
    all_tables = []
    all_images = []

    for doc in documents:
        LOGGER.info("Parsing document: %s", doc)
        text_objs, table_objs, image_objs = ingest_document(doc)
        all_texts.extend(text_objs)
        all_tables.extend(table_objs)
        all_images.extend(image_objs)

    return all_texts, all_images, all_tables


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.CRITICAL, format="%(name)s %(asctime)s %(message)s"
    )
    LOGGER.setLevel(logging.INFO)

    folder = Path(args.document_folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(
            f"Document folder does not exist or is not a directory: {folder}"
        )

    LOGGER.info("Parsing supported documents in %s", folder)
    all_documents = get_documents(str(folder))
    all_texts, all_images, all_tables = process_documents(all_documents)

    LOGGER.info("Generating descriptions and summaries for objects")
    all_texts, all_images, all_tables = summarize_objects(
        all_texts, all_images, all_tables
    )

    LOGGER.info("Generating retriever and DB")
    retriever = generate_database_and_retriever(main_folder=args.data_base)

    LOGGER.info("Populating DB")
    retriever = populate_database(retriever, all_texts, all_images, all_tables)

    LOGGER.info("Successfully populated the DB")


if __name__ == "__main__":
    main()
