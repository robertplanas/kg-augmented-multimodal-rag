import hashlib
import os
import chromadb
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

import json

import logging
from utils.models import EmbeddingModel

LOGGER = logging.getLogger(__name__)


def flatten_metadata(metadata_dict):
    """
    Converts nested lists/dicts into JSON strings so Vector DBs can accept them.
    """
    flattened = {}
    for key, value in metadata_dict.items():
        if isinstance(value, (dict, list)):
            # Turn the complex list/dict into a JSON string
            flattened[key] = json.dumps(value)
        else:
            flattened[key] = value
    return flattened


def add_to_retriever(
    retriever, id_key, objects, attr_name, type_val
):  # renamed 'type' to avoid shadowing

    existing_keys = set(retriever.docstore.yield_keys())

    new_summaries = []  # These go to Vector Store
    new_store_records = []  # These go to Doc Store
    new_ids = []

    for i, obj in enumerate(objects):
        content = getattr(obj, attr_name)
        metadata = getattr(obj, "metadata", {})
        id = generate_id(
            getattr(obj, attr_name),
            metadata.get("filename", "unknown"),
        )

        if id not in existing_keys:
            new_ids.append(id)

            # 1. The Summary Document (for Vector Store)
            new_summaries.append(
                Document(
                    page_content=obj.description,
                    metadata={
                        id_key: id,
                        "type": type_val,
                        "metadata": json.dumps(metadata),
                    },
                )
            )

            # 2. The Wrapped Document (for Doc Store)
            record = {
                "content": content,
                "id": id,
                "context": getattr(obj, "context", ""),
                "description": getattr(obj, "description", ""),
                "type": type_val,
                "metadata": json.dumps(metadata),
            }
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
    return hashlib.sha256((filename + "_" + content).encode("utf-8")).hexdigest()


def generate_node_id(name: str, type: str, description: str):
    if name is None:
        name = ""
    if type is None:
        type = ""
    if description is None:
        description = ""
    return hashlib.sha256(
        (name + "_" + type + "_" + description).encode("utf-8")
    ).hexdigest()


def generate_database_and_retriever(
    chroma_index_folder="chroma_index",
    raw_data_folder="raw_data",
    main_folder="./localdb",
    db_name="multi_modal_rag",
    ollama_model_name="embeddinggemma:latest",
    embedding_provider="ollama",
    embedding_model_name=None,
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

    model_name = embedding_model_name or ollama_model_name
    embedding_model = EmbeddingModel(
        model_name=model_name,
        provider=embedding_provider,
    )

    vectorstore = Chroma(
        client=client,
        collection_name=db_name,
        embedding_function=embedding_model.as_langchain_embedding(),
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


def populate_community_database(
    retriever,
    communities,
    level_type,  # e.g., "global" or "mid"
    id_key="doc_id",
):
    """
    Injisents KG community reports into the MultiVectorRetriever.
    'communities' is a list of objects with summary, full_report, and metadata.
    """
    existing_keys = set(retriever.docstore.yield_keys())

    new_summaries = []  # For Vector Store (semantic search)
    new_docstore_payloads = []  # For Doc Store (LLM context)
    new_ids = []

    for community in communities:
        # Create a unique ID based on the level and community ID
        # Example: 'global_community_42'
        comm_id = f"{level_type}_{getattr(community, 'id', 'unknown')}"

        if comm_id not in existing_keys:
            # 1. Prepare Summary Document for Vector Search
            # We search against the summary because it's densly packed with keywords
            summary_doc = Document(
                page_content=community.summary,
                metadata={
                    id_key: comm_id,
                    "level": level_type,
                    "title": getattr(community, "title", ""),
                },
            )

            # 2. Prepare Full Report for Docstore
            # This is what the retriever returns to the LLM
            full_report = {
                "community_id": comm_id,
                "level": level_type,
                "summary": community.summary,
                "full_content": community.full_report,  # Detailed entities/edges
            }

            new_ids.append(comm_id)
            new_summaries.append(summary_doc)
            new_docstore_payloads.append(json.dumps(full_report).encode("utf-8"))

    if new_summaries:
        # Step A: Add to Vector Store
        retriever.vectorstore.add_documents(new_summaries)
        # Step B: Add to Doc Store
        retriever.docstore.mset(list(zip(new_ids, new_docstore_payloads)))
        print(f"Added {len(new_ids)} {level_type} communities to the retriever.")
    else:
        print(f"No new {level_type} communities to add.")


def populate_node_db(
    retriever,
    nodes,
):

    new_nodes = []  # For Vector Store (semantic search)
    new_docstore_payloads = []  # For Doc Store (LLM context)
    new_ids = []

    id_key = retriever.id_key

    existing_keys = set(retriever.docstore.yield_keys())
    for node in nodes:
        node_id = generate_node_id(node["name"], node["type"], node["description"])
        if node_id not in existing_keys:
            node_doc = Document(
                page_content=node["name"],
                metadata={
                    id_key: node_id,
                },
            )

            full_node_report = {
                "node_id": node_id,
                "name": node["name"],
                "type": node["type"],
                "description": node["description"],
            }
            new_ids.append(node_id)
            new_nodes.append(node_doc)
            new_docstore_payloads.append(json.dumps(full_node_report).encode("utf-8"))

    if new_nodes:
        # Step A: Add to Vector Store
        retriever.vectorstore.add_documents(new_nodes)
        # Step B: Add to Doc Store
        retriever.docstore.mset(list(zip(new_ids, new_docstore_payloads)))
        print(f"Added {len(new_ids)} nodes to the retriever.")
    else:
        print("No new nodes to add.")
