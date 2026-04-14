import argparse
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from graphdatascience import GraphDataScience
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase

from utils.database_utils import (
    generate_database_and_retriever,
    populate_community_database,
    populate_node_db,
)
from utils.models import LLMModel


LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()

PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def _load_project_env() -> Path:
    project_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=project_root / ".env")
    load_dotenv(dotenv_path=project_root / ".venv" / ".env")
    return project_root

SYSTEM_INSTRUCTION = """
You are an expert Graph Data Analyst. Your task is to synthesize information about a specific community of entities into a structured report.

### Instructions
1. **Title**: Create a concise, descriptive title that captures the community's essence.
2. **Summary**: Provide a high-level overview (1-2 sentences) of the core theme and why these entities are grouped together.
4. **Relationship Dynamics**: Describe the primary interactions. How do these entities support, compete with, or interact with one another?

### Output Format
Return your response in clean Markdown. Use headers for sections and bold text for entity names. Do not include any conversational filler.
"""

PROMPT = """
### Community Data
**Nodes within this community:**
{nodes}

**Relationships/Edges:**
{relationships}

Please generate the community report based on the System Instructions.
"""


@dataclass
class KGCommunity:
    id: int
    title: str
    summary: str
    full_report: str
    nodes: str
    relationships: str
    metadata: dict = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find graph communities and populate community DBs")

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
        "--community_graph_name",
        type=str,
        default="my-community-graph",
        help="In-memory GDS graph projection name.",
    )
    parser.add_argument(
        "--excluded_labels",
        nargs="+",
        default=["image", "text", "table", "file", "code"],
        help="Node labels excluded from community graph projection.",
    )
    parser.add_argument(
        "--global_gamma",
        type=float,
        default=0.2,
        help="Gamma parameter for global community Leiden run.",
    )
    parser.add_argument(
        "--mid_gamma",
        type=float,
        default=1.5,
        help="Gamma parameter for mid-level community Leiden run.",
    )
    parser.add_argument(
        "--global_property",
        type=str,
        default="globalCommunityId",
        help="Node property used to write global communities.",
    )
    parser.add_argument(
        "--mid_property",
        type=str,
        default="midCommunityId",
        help="Node property used to write mid-level communities.",
    )

    parser.add_argument(
        "--summary_provider",
        type=str,
        default=None,
        help="Community summary LLM provider override (e.g., ollama, openai, google).",
    )
    parser.add_argument(
        "--summary_model",
        type=str,
        default=None,
        help="Community summary LLM model override.",
    )
    parser.add_argument(
        "--summary_max_workers",
        type=int,
        default=None,
        help=(
            "Community summary worker count override. "
            "If omitted, uses COMMUNITY_SUMMARY_MAX_WORKERS/SUMMARY_LLM_MAX_WORKERS or default."
        ),
    )

    parser.add_argument(
        "--embedding_provider",
        type=str,
        default="ollama",
        help="Embedding provider for community/node retrievers.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="embeddinggemma:latest",
        help="Embedding model for community/node retrievers.",
    )

    parser.add_argument(
        "--global_communities_db",
        type=str,
        default="./localdb/global_communities",
        help="Folder for global communities retriever DB.",
    )
    parser.add_argument(
        "--mid_communities_db",
        type=str,
        default="./localdb/mid_communities",
        help="Folder for mid communities retriever DB.",
    )
    parser.add_argument(
        "--node_db",
        type=str,
        default="./localdb/node_db",
        help="Folder for node retriever DB.",
    )

    return parser.parse_args()


def _resolve_provider_api_kwargs(provider: str, step_name: str):
    project_root = _load_project_env()
    resolved_provider = (provider or "ollama").lower()
    api_key_env = PROVIDER_API_KEY_ENV.get(resolved_provider)
    if not api_key_env:
        return {}

    api_key = os.getenv(api_key_env)
    if api_key:
        return {"api_key": api_key}

    raise ValueError(
        f"Missing API key for {step_name} provider '{resolved_provider}'. "
        f"Set {api_key_env} in your environment or in {project_root / '.env'}."
    )


def _resolve_summary_config(provider, model_name, max_workers):
    _load_project_env()

    resolved_provider = (
        provider
        or os.getenv("COMMUNITY_SUMMARY_PROVIDER")
        or os.getenv("SUMMARY_LLM_PROVIDER")
        or "ollama"
    ).lower()
    resolved_model_name = (
        model_name
        or os.getenv("COMMUNITY_SUMMARY_MODEL")
        or os.getenv("SUMMARY_LLM_MODEL")
        or "gemma3:12b"
    )

    llm_kwargs = _resolve_provider_api_kwargs(
        resolved_provider,
        "community summary",
    )

    if max_workers is None:
        env_workers = os.getenv("COMMUNITY_SUMMARY_MAX_WORKERS") or os.getenv(
            "SUMMARY_LLM_MAX_WORKERS"
        )
        env_workers = env_workers or "1"
        try:
            resolved_max_workers = int(env_workers)
        except ValueError:
            raise ValueError(
                "Invalid community summary worker configuration value: "
                f"{env_workers}. Expected integer."
            ) from None
    else:
        resolved_max_workers = max_workers

    if resolved_max_workers < 1:
        raise ValueError("Community summary max workers must be >= 1.")

    return resolved_provider, resolved_model_name, llm_kwargs, resolved_max_workers


def _resolve_neo4j_auth(args: argparse.Namespace) -> tuple[str, str]:
    _load_project_env()
    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError(
            "Missing Neo4j password. Set --neo4j_password or NEO4J_PASSWORD env var."
        )
    return args.neo4j_user, password


def _fetch_projection_config(gds: GraphDataScience, excluded_labels: list[str]):
    all_labels = gds.run_cypher("CALL db.labels()")
    labels = all_labels["label"].tolist()
    excluded = {str(label) for label in excluded_labels}
    labels = [label for label in labels if label not in excluded]

    if not labels:
        raise ValueError(
            "No labels available for projection after exclusions. "
            f"Excluded labels: {sorted(excluded)}"
        )

    all_rels = gds.run_cypher("CALL db.relationshipTypes()")
    relationship_types = all_rels["relationshipType"].tolist()
    if not relationship_types:
        raise ValueError("No relationship types found in Neo4j.")

    rel_config = {
        relation_type: {"orientation": "UNDIRECTED"}
        for relation_type in relationship_types
    }

    return labels, rel_config


def _project_graph(gds: GraphDataScience, graph_name: str, labels, rel_config):
    LOGGER.info(
        "Projecting GDS graph '%s' with %d labels and %d relationship types.",
        graph_name,
        len(labels),
        len(rel_config),
    )

    try:
        graph, _ = gds.graph.project(
            graph_name,
            labels,
            rel_config,
        )
        return graph
    except Exception:
        LOGGER.warning(
            "Projection failed for '%s'. Attempting to drop any pre-existing in-memory graph and retry.",
            graph_name,
        )
        try:
            gds.graph.drop(graph_name)
        except Exception:
            escaped_name = graph_name.replace("'", "\\'")
            gds.run_cypher(
                f"CALL gds.graph.drop('{escaped_name}', false) YIELD graphName RETURN graphName"
            )

        graph, _ = gds.graph.project(
            graph_name,
            labels,
            rel_config,
        )
        return graph


def _run_leiden(gds: GraphDataScience, graph, args: argparse.Namespace) -> None:
    LOGGER.info(
        "Running Leiden for global communities: writeProperty=%s gamma=%.4f",
        args.global_property,
        args.global_gamma,
    )
    gds.leiden.write(
        graph,
        writeProperty=args.global_property,
        includeIntermediateCommunities=False,
        relationshipWeightProperty=None,
        gamma=args.global_gamma,
    )

    LOGGER.info(
        "Running Leiden for mid communities: writeProperty=%s gamma=%.4f",
        args.mid_property,
        args.mid_gamma,
    )
    gds.leiden.write(
        graph,
        writeProperty=args.mid_property,
        includeIntermediateCommunities=False,
        relationshipWeightProperty=None,
        gamma=args.mid_gamma,
    )


def fetch_community_data(driver, database: str, level: str = "global"):
    if level not in {"global", "mid"}:
        raise ValueError("level must be either 'global' or 'mid'")

    query = f"""
    MATCH (n)
    WHERE n.{level}CommunityId IS NOT NULL
    WITH n.{level}CommunityId AS communityId, collect(n) AS nodes
    UNWIND nodes AS source
    MATCH (source)-[r]->(target)
    WHERE target IN nodes
    RETURN communityId,
           [node IN nodes | node.name] AS entity_names,
           collect({{s: source.name, t: type(r), o: target.name}}) AS triples
    """

    community_inputs = {}
    with driver.session(database=database) as session:
        records = session.run(query)

        for record in records:
            entity_names = [
                str(name) for name in (record["entity_names"] or []) if name is not None
            ]
            entities_str = ", ".join(entity_names)
            triples_str = "\n".join(
                [
                    f"- {triple.get('s')} --[{triple.get('t')}]--> {triple.get('o')}"
                    for triple in (record["triples"] or [])
                ]
            )

            community_report_input = {
                "community_id": record["communityId"],
                "nodes": entities_str,
                "relationships": triples_str,
            }
            community_inputs[record["communityId"]] = community_report_input

    return community_inputs


def fetch_nodes_data(driver, database: str):
    query = """
    MATCH (n)
    WHERE NOT any(label IN labels(n) WHERE label IN ["file", "text", "image", "table", "code"])
    RETURN
        n.name AS name,
        labels(n)[0] AS type,
        n.description AS description
    """

    with driver.session(database=database) as session:
        result = session.run(query)
        nodes = [
            {
                "name": record["name"],
                "type": record["type"],
                "description": record["description"],
            }
            for record in result
        ]

    return nodes


class CommunitySummarizer:
    def __init__(
        self,
        model_name: str,
        provider: str,
        **llm_kwargs,
    ):
        self.model_name = model_name
        self.provider = provider
        self.llm_kwargs = llm_kwargs

    def _build_chain(self):
        llm = LLMModel(
            model_name=self.model_name,
            provider=self.provider,
            temperature=0,
            **self.llm_kwargs,
        ).as_langchain_llm()
        return llm | StrOutputParser()

    def _get_thread_chain(self):
        key = (
            self.model_name,
            self.provider,
            tuple(sorted(self.llm_kwargs.items())),
        )
        if getattr(_THREAD_LOCAL, "community_chain_key", None) != key:
            _THREAD_LOCAL.community_chain = self._build_chain()
            _THREAD_LOCAL.community_chain_key = key
        return _THREAD_LOCAL.community_chain

    def _build_messages(self, community_data: dict):
        nodes = community_data.get("nodes", "No node data available.")
        relationships = community_data.get(
            "relationships", "No relationship data available."
        )
        formatted_prompt = PROMPT.format(
            nodes=nodes,
            relationships=relationships,
        )
        return [
            SystemMessage(content=SYSTEM_INSTRUCTION),
            HumanMessage(content=formatted_prompt),
        ]

    def _summarize_single(self, community_id, community_data):
        chain = self._get_thread_chain()
        messages = self._build_messages(community_data)
        LOGGER.info("Generating summary for community %s", community_id)
        summary = chain.invoke(messages)
        return community_id, summary

    def find_community_summary(
        self,
        community_inputs: dict,
        max_workers: int = 1,
        execution_mode: str = "thread",
    ) -> None:
        if not community_inputs:
            return

        if execution_mode != "thread" or max_workers <= 1 or len(community_inputs) <= 1:
            chain = self._build_chain()
            for community_id, community_data in community_inputs.items():
                messages = self._build_messages(community_data)
                LOGGER.info("Generating summary for community %s", community_id)
                summary = chain.invoke(messages)
                community_inputs[community_id]["summary"] = summary
            return

        LOGGER.info(
            "Generating summaries for %d communities using %d workers.",
            len(community_inputs),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_community_id = {
                executor.submit(self._summarize_single, community_id, community_data): community_id
                for community_id, community_data in community_inputs.items()
            }
            for future in as_completed(future_to_community_id):
                community_id = future_to_community_id[future]
                _, summary = future.result()
                community_inputs[community_id]["summary"] = summary


def transform_dicts_to_objects(community_dict: dict) -> list[KGCommunity]:
    community_objects = []

    for community_id, data in community_dict.items():
        summary_lines = data.get("summary", "").split("\n")
        title = (
            summary_lines[0].replace("#", "").strip()
            if summary_lines and summary_lines[0].strip()
            else f"Community {community_id}"
        )

        full_report = (
            f"COMMUNITY NODES: {data.get('nodes')}\n\n"
            f"RELATIONSHIPS:\n{data.get('relationships')}\n\n"
            f"{data.get('summary')}"
        )

        obj = KGCommunity(
            id=int(community_id),
            title=title,
            summary=data.get("summary", ""),
            full_report=full_report,
            nodes=data.get("nodes", ""),
            relationships=data.get("relationships", ""),
            metadata={"source": "kg_extraction", "community_id": int(community_id)},
        )
        community_objects.append(obj)

    return community_objects


def _create_retrievers(args: argparse.Namespace):
    global_retriever = generate_database_and_retriever(
        main_folder=args.global_communities_db,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )
    mid_retriever = generate_database_and_retriever(
        main_folder=args.mid_communities_db,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )
    node_retriever = generate_database_and_retriever(
        main_folder=args.node_db,
        db_name="node_db",
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )
    return global_retriever, mid_retriever, node_retriever


def _run_pipeline() -> None:
    args = parse_args()

    if args.global_gamma < 0:
        raise ValueError("--global_gamma must be >= 0")
    if args.mid_gamma < 0:
        raise ValueError("--mid_gamma must be >= 0")

    summary_provider, summary_model, summary_kwargs, summary_max_workers = _resolve_summary_config(
        provider=args.summary_provider,
        model_name=args.summary_model,
        max_workers=args.summary_max_workers,
    )
    auth = _resolve_neo4j_auth(args)

    LOGGER.info("Connecting to Neo4j and GDS.")
    gds = GraphDataScience(
        args.neo4j_uri,
        auth=auth,
        database=args.neo4j_database,
    )

    graph = None
    try:
        labels, rel_config = _fetch_projection_config(gds, args.excluded_labels)
        graph = _project_graph(gds, args.community_graph_name, labels, rel_config)
        _run_leiden(gds, graph, args)
    finally:
        if graph is not None:
            LOGGER.info("Dropping in-memory GDS graph '%s'.", args.community_graph_name)
            gds.graph.drop(graph)

    LOGGER.info("Fetching communities and nodes from Neo4j.")
    with GraphDatabase.driver(args.neo4j_uri, auth=auth) as driver:
        global_community_inputs = fetch_community_data(
            driver,
            database=args.neo4j_database,
            level="global",
        )
        mid_community_inputs = fetch_community_data(
            driver,
            database=args.neo4j_database,
            level="mid",
        )
        nodes = fetch_nodes_data(driver, database=args.neo4j_database)

    LOGGER.info(
        "Fetched %d global communities, %d mid communities, and %d nodes.",
        len(global_community_inputs),
        len(mid_community_inputs),
        len(nodes),
    )

    community_summarizer = CommunitySummarizer(
        model_name=summary_model,
        provider=summary_provider,
        **summary_kwargs,
    )

    LOGGER.info("Generating summaries for global communities.")
    community_summarizer.find_community_summary(
        global_community_inputs,
        max_workers=summary_max_workers,
        execution_mode="thread",
    )
    LOGGER.info("Generating summaries for mid communities.")
    community_summarizer.find_community_summary(
        mid_community_inputs,
        max_workers=summary_max_workers,
        execution_mode="thread",
    )

    global_communities_objects = transform_dicts_to_objects(global_community_inputs)
    mid_community_objects = transform_dicts_to_objects(mid_community_inputs)

    global_retriever, mid_retriever, node_retriever = _create_retrievers(args)

    LOGGER.info("Populating global communities DB.")
    populate_community_database(global_retriever, global_communities_objects, "global")

    LOGGER.info("Populating mid communities DB.")
    populate_community_database(mid_retriever, mid_community_objects, "mid")

    LOGGER.info("Populating node DB.")
    populate_node_db(node_retriever, nodes)



def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(asctime)s %(message)s")
    _run_pipeline()


if __name__ == "__main__":
    main()
