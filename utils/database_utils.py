import hashlib
import os
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

import json

import logging

LOGGER = logging.getLogger(__name__)


def add_to_retriever(
    retriever, id_key, objects, attr_name, type_val
):  # renamed 'type' to avoid shadowing
    payloads = [getattr(obj, attr_name) for obj in objects]

    ids = [generate_id(p) for p in payloads]

    existing_keys = set(retriever.docstore.yield_keys())

    new_summaries = []  # These go to Vector Store
    new_store_records = []  # These go to Doc Store
    new_ids = []

    for i, obj in enumerate(objects):
        if ids[i] not in existing_keys:
            new_ids.append(ids[i])

            # 1. The Summary Document (for Vector Store)
            new_summaries.append(
                Document(
                    page_content=obj.description,
                    metadata={id_key: ids[i], "type": type_val},
                )
            )

            record = {"content": payloads[i], "type": type_val}
            new_store_records.append(json.dumps(record).encode("utf-8"))

    if new_summaries:
        # Add summaries to Vector Store
        retriever.vectorstore.add_documents(new_summaries)

        # Add wrapped Documents to Doc Store
        retriever.docstore.mset(list(zip(new_ids, new_store_records)))
        LOGGER.info(f"Added {len(new_ids)} new items for {attr_name}.")
    else:
        LOGGER.info(f"No new items to add for {attr_name}.")


def generate_id(filename: str, content: str):
    """Generates a unique, stable hex ID based on content and context."""
    return hashlib.sha256(
        filename.encode("utf-8") + "_" + content.encode("utf-8")
    ).hexdigest()


def generate_database_and_retriever(
    chroma_index_folder="chroma_index",
    raw_data_folder="raw_data",
    main_folder="./localdb",
    db_name="multi_modal_rag",
    ollama_model_name="embeddinggemma:latest",
):
    if not os.path.exists(main_folder):
        LOGGER.info("Creating local db folder.")
        os.makedirs(main_folder)

    else:
        LOGGER.info("Local db folder already exists.")

    chroma_index_folder_complete = os.path.join(main_folder, chroma_index_folder)
    if not os.path.exists(chroma_index_folder_complete):
        LOGGER.info("Creating Chroma index folder.")
        os.makedirs(chroma_index_folder_complete)
    else:
        LOGGER.info("Chroma index folder already exists.")

    raw_data_folder_complete = os.path.join(main_folder, raw_data_folder)
    if not os.path.exists(raw_data_folder_complete):
        LOGGER.info("Creating Raw data folder.")
        os.makedirs(raw_data_folder_complete)
    else:
        LOGGER.info("Raw data folder already exists.")

    client = chromadb.PersistentClient(path=chroma_index_folder_complete)

    LOGGER.info("Initiating Vector Store")

    vectorstore = Chroma(
        client=client,
        collection_name=db_name,
        embedding_function=OllamaEmbeddings(model=ollama_model_name),
    )

    store = LocalFileStore(raw_data_folder_complete)
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=store, id_key=id_key
    )

    return retriever


def populate_database(retriever, text_objs, images_obj, tables_obj):
    LOGGER.info("Adding data to persistent db")
    add_to_retriever(retriever, "doc_id", text_objs, "text", "text")
    add_to_retriever(retriever, "doc_id", tables_obj, "markdown", "table")
    add_to_retriever(retriever, "doc_id", images_obj, "base64", "image")

    LOGGER.info("All data persisted to Local Database")
    return retriever
