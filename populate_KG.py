import argparse
import asyncio
import logging
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv

from utils.database_utils import generate_database_and_retriever
from utils.knowledge_graph import (
    convert_to_graph_elements_pipeline,
    fetch_existing_neo4j_nodes,
)
from utils.node_standarization import (
    apply_representatives_to_relationships,
    build_nodes_from_relationships,
    build_representatives,
    group_nodes_by_similarity,
    lemmatize_nodes_and_relationships,
    translate_nodes,
)


LOGGER = logging.getLogger(__name__)

PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate KG")
    parser.add_argument(
        "-db",
        "--data_base",
        type=str,
        default="./localdb",
        help="Folder where the DB will be stored.",
    )

    parser.add_argument(
        "--KG_extractor_provider",
        "--kg_provider",
        dest="kg_provider",
        type=str,
        default=None,
        help=(
            "KG extractor LLM provider override (e.g., ollama, openai, google). "
            "If omitted, uses KG_EXTRACTOR_PROVIDER/KG_LLM_PROVIDER or default."
        ),
    )
    parser.add_argument(
        "--KG_extractor_model",
        "--kg_model",
        dest="kg_model",
        type=str,
        default=None,
        help=(
            "KG extractor LLM model override. "
            "If omitted, uses KG_EXTRACTOR_MODEL/KG_LLM_MODEL or default."
        ),
    )
    parser.add_argument(
        "--KG_max_workers",
        "--kg_max_workers",
        dest="kg_max_workers",
        type=int,
        default=None,
        help=(
            "KG worker count override. "
            "If omitted, uses KG_MAX_WORKERS/KG_LLM_MAX_WORKERS or default."
        ),
    )

    parser.add_argument(
        "--node_translate_provider",
        type=str,
        default=None,
        help="Provider for node translation step.",
    )
    parser.add_argument(
        "--node_translate_model",
        type=str,
        default="gemma3:latest",
        help="Model for node translation step.",
    )
    parser.add_argument(
        "--node_translate_max_workers",
        type=int,
        default=None,
        help="Workers for node translation step.",
    )
    parser.add_argument(
        "--node_translate_batch_size",
        type=int,
        default=30,
        help="Batch size for node translation step.",
    )

    parser.add_argument(
        "--node_embedding_provider",
        type=str,
        default="ollama",
        help="Embedding provider used for node grouping.",
    )
    parser.add_argument(
        "--node_embedding_model",
        type=str,
        default="embeddinggemma:latest",
        help="Embedding model used for node grouping.",
    )

    parser.add_argument(
        "--node_description_provider",
        type=str,
        default="ollama",
        help="Provider used to summarize representative node descriptions.",
    )
    parser.add_argument(
        "--node_description_model",
        type=str,
        default="gemma3:12b",
        help="Model used to summarize representative node descriptions.",
    )

    parser.add_argument(
        "--group_threshold",
        type=float,
        default=0.05,
        help="Cosine-distance threshold for grouping extracted nodes.",
    )
    parser.add_argument(
        "--existing_node_threshold",
        type=float,
        default=0.05,
        help="Cosine-distance threshold for matching groups to existing Neo4j nodes.",
    )

    parser.add_argument(
        "--reuse_existing_nodes",
        action="store_true",
        help="Reuse existing Neo4j nodes if close matches are found.",
    )
    parser.add_argument(
        "--neo4j_uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI.",
    )
    parser.add_argument(
        "--neo4j_user",
        type=str,
        default="neo4j",
        help="Neo4j user.",
    )
    parser.add_argument(
        "--neo4j_password",
        type=str,
        default=None,
        help="Neo4j password. If omitted, uses NEO4J_PASSWORD env var.",
    )
    parser.add_argument(
        "--neo4j_database",
        type=str,
        default="neo4j",
        help="Neo4j database name.",
    )

    parser.add_argument(
        "--raw_output_file",
        type=str,
        default=None,
        help="Optional Path to store raw extracted relationships.",
    )
    parser.add_argument(
        "--translated_output_file",
        type=str,
        default=None,
        help="Optional path to store translated relationships.",
    )
    parser.add_argument(
        "--lemmatized_output_file",
        type=str,
        default=None,
        help="Optional path to store lemmatized relationships.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional Path to store final cleaned relationships.",
    )

    return parser.parse_args()


def _resolve_kg_config(provider, model_name, max_workers):
    project_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=project_root / ".env")

    resolved_provider = (
        provider
        or os.getenv("KG_EXTRACTOR_PROVIDER")
        or os.getenv("KG_LLM_PROVIDER")
        or "ollama"
    ).lower()
    resolved_model_name = (
        model_name
        or os.getenv("KG_EXTRACTOR_MODEL")
        or os.getenv("KG_LLM_MODEL")
        or "gemma3:12b"
    )

    if max_workers is None:
        env_workers = os.getenv("KG_MAX_WORKERS") or os.getenv("KG_LLM_MAX_WORKERS")
        env_workers = env_workers or "1"
        try:
            resolved_max_workers = int(env_workers)
        except ValueError:
            raise ValueError(
                f"Invalid KG worker configuration value: {env_workers}. Expected integer."
            ) from None
    else:
        resolved_max_workers = max_workers

    if resolved_max_workers < 1:
        raise ValueError("KG max workers must be >= 1.")

    api_key_env = PROVIDER_API_KEY_ENV.get(resolved_provider)
    if api_key_env and not os.getenv(api_key_env):
        raise ValueError(
            f"Missing API key for KG provider '{resolved_provider}'. "
            f"Set {api_key_env} in your environment or in {project_root / '.env'}."
        )

    return resolved_provider, resolved_model_name, resolved_max_workers


def _resolve_provider_api_kwargs(provider: str, step_name: str):
    resolved_provider = (provider or "ollama").lower()
    api_key_env = PROVIDER_API_KEY_ENV.get(resolved_provider)
    if not api_key_env:
        return {}

    api_key = os.getenv(api_key_env)
    if api_key:
        return {"api_key": api_key}

    raise ValueError(
        f"Missing API key for {step_name} provider '{resolved_provider}'. "
        f"Set {api_key_env} in your environment or in {Path(__file__).resolve().parent / '.env'}."
    )


def _save_pickle(payload, output_file: str) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def _get_existing_nodes_for_reuse(args: argparse.Namespace):
    if not args.reuse_existing_nodes:
        return []

    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError(
            "--reuse_existing_nodes requires --neo4j_password or NEO4J_PASSWORD env var."
        )

    LOGGER.info(
        "Loading existing nodes from Neo4j %s (database=%s)",
        args.neo4j_uri,
        args.neo4j_database,
    )
    existing_nodes = fetch_existing_neo4j_nodes(
        uri=args.neo4j_uri,
        auth=(args.neo4j_user, password),
        database=args.neo4j_database,
    )
    LOGGER.info("Fetched %d existing Neo4j nodes for reuse.", len(existing_nodes))
    return existing_nodes


async def _run_kg_pipeline() -> None:
    args = parse_args()

    if args.group_threshold < 0:
        raise ValueError("--group_threshold must be >= 0")
    if args.existing_node_threshold < 0:
        raise ValueError("--existing_node_threshold must be >= 0")
    if args.node_translate_batch_size < 1:
        raise ValueError("--node_translate_batch_size must be >= 1")

    provider, model_name, max_workers = _resolve_kg_config(
        provider=args.kg_provider,
        model_name=args.kg_model,
        max_workers=args.kg_max_workers,
    )

    retriever = generate_database_and_retriever(main_folder=args.data_base)
    all_keys = list(retriever.docstore.yield_keys())
    all_documents = retriever.docstore.mget(all_keys)
    documents_dict = {all_keys[i]: all_documents[i] for i in range(len(all_keys))}

    if not documents_dict:
        LOGGER.info("No documents found in '%s'. Nothing to process.", args.data_base)
        return

    LOGGER.info(
        "Extracting KG relationships for %d documents with provider=%s model=%s workers=%d",
        len(documents_dict),
        provider,
        model_name,
        max_workers,
    )

    graph_data = await convert_to_graph_elements_pipeline(
        documents_dict,
        model_name_text=model_name,
        model_name_table=model_name,
        model_name_image=model_name,
        model_name_code=model_name,
        provider_text=provider,
        provider_table=provider,
        provider_image=provider,
        max_concurrency=max_workers,
        execution_mode="thread",
    )

    LOGGER.info("Finished KG extraction for %d documents.", len(graph_data))
    if args.raw_output_file is not None:
        _save_pickle(graph_data, args.raw_output_file)
        LOGGER.info("Saved raw graph data to %s", args.raw_output_file)

    LOGGER.info("Translating nodes.")
    graph_translated = translate_nodes(
        graph_data,
        model_translate=args.node_translate_model,
        provider=args.node_translate_provider,
        max_workers=args.node_translate_max_workers,
        batch_size=args.node_translate_batch_size,
        execution_mode="thread",
    )

    if args.translated_output_file:
        _save_pickle(graph_translated, args.translated_output_file)
        LOGGER.info("Saved translated graph data to %s", args.translated_output_file)

    LOGGER.info("Lemmatizing nodes and node types.")
    graph_lemmatized = lemmatize_nodes_and_relationships(graph_translated)

    if args.lemmatized_output_file:
        _save_pickle(graph_lemmatized, args.lemmatized_output_file)
        LOGGER.info("Saved lemmatized graph data to %s", args.lemmatized_output_file)

    existing_nodes = _get_existing_nodes_for_reuse(args)
    node_embedding_kwargs = _resolve_provider_api_kwargs(
        args.node_embedding_provider,
        "node embedding",
    )
    node_description_kwargs = _resolve_provider_api_kwargs(
        args.node_description_provider,
        "node description",
    )

    LOGGER.info("Building node embeddings for canonicalization.")
    all_nodes = build_nodes_from_relationships(
        graph_lemmatized,
        embedding_model_name=args.node_embedding_model,
        embedding_provider=args.node_embedding_provider,
        **node_embedding_kwargs,
    )
    LOGGER.info("Built %d node instances from extracted relationships.", len(all_nodes))

    nodes_by_group = group_nodes_by_similarity(
        all_nodes,
        threshold=args.group_threshold,
    )
    LOGGER.info(
        "Created %d node groups with threshold %.4f.",
        len(nodes_by_group),
        args.group_threshold,
    )

    representatives = build_representatives(
        nodes_by_group,
        existing_nodes=existing_nodes,
        existing_node_threshold=args.existing_node_threshold,
        embedding_model_name=args.node_embedding_model,
        embedding_provider=args.node_embedding_provider,
        description_model_name=args.node_description_model,
        description_provider=args.node_description_provider,
        embedding_kwargs=node_embedding_kwargs,
        description_llm_kwargs=node_description_kwargs,
    )
    LOGGER.info("Created %d representative nodes.", len(representatives))

    graph_clean = apply_representatives_to_relationships(
        graph_lemmatized,
        representatives,
    )

    _save_pickle(graph_clean, args.output_file)
    LOGGER.info("Saved cleaned graph data to %s", args.output_file)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(asctime)s %(message)s")
    asyncio.run(_run_kg_pipeline())


if __name__ == "__main__":
    main()
