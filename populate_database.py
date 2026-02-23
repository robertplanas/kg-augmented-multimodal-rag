import argparse
import logging
import glob

from utils.ingest import ingest_document
from utils.database_utils import generate_database_and_retriever, populate_database
from utils.summarize import summarize_objects


LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Populate DB")

parser.add_argument(
    "-df",
    "--document_folder",
    type=str,
    default=None,
    help="Folder where the documents that need parsing are stored.",
    required=True,
)

parser.add_argument(
    "-db",
    "--data_base",
    type=str,
    default="./localdb",
    help="Folder where the DB will be stored.",
    required=True,
)

args = parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.CRITICAL, format="%(name)s %(asctime)s %(message)s"
    )
    LOGGER.setLevel(logging.INFO)

    folder = args.document_folder
    assert folder is not None and folder != "", "Folder needs to be specified"
    LOGGER.info("Parsing all pdf documents in {}".format(folder))
    all_documents = glob.glob(f"{folder}/*.pdf")

    all_texts = []
    all_tables = []
    all_images = []

    for doc in all_documents:
        LOGGER.info("Parsing document: {}".format(doc))
        text_objs, table_objs, images_objs = ingest_document(doc)
        all_texts.extend(text_objs)
        all_tables.extend(table_objs)
        all_images.extend(images_objs)

    LOGGER.info("Generating descriptions and summaries for objects")
    all_texts, all_images, all_tables = summarize_objects(
        all_texts, all_images, all_tables
    )

    LOGGER.info("Generating retriever and DB")
    retriever = generate_database_and_retriever(main_folder=args.data_base)

    LOGGER.info("Populating DB")
    retriever = populate_database(retriever, all_texts, all_images, all_tables)

    LOGGER.info("Sucessfully populated the DB")
