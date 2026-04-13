import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase

from utils.database_utils import generate_database_and_retriever
from utils.models import LLMModel
from utils.rag import parse_documents


LOGGER = logging.getLogger(__name__)

PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

RAG_SYSTEM_PROMPT = """
You are an expert analytical assistant. Your task is to provide a comprehensive and accurate answer to the user's query based strictly on the provided context.
INSTRUCTIONS:
1. Synthesize the answer by starting with the specific facts (<Specific_Documents> and <Visual_Context>).
2. Use the <Entity_Nodes> to clarify any relationships between actors or concepts.
3. Weave in the <Mid_Level_Context> and <Global_Context> to provide background, explain the "why", or fill in gaps if the specific documents are sparse.
4. If the different contexts contradict each other, prioritize <Specific_Documents> for exact facts, but note the discrepancy in your answer.
5. Do not include information that is not present in the contexts.
"""

RAG_USER_PROMPT = """
The context is derived from multiple sources: high-level summaries, specific entity data, raw document snippets, and visual inputs.

<Global_Context>
[Use this to understand the overarching environment or dataset as a whole]
{summaries_global_level_communities}
</Global_Context>

<Mid_Level_Context>
[Use this to understand thematic clusters, regional trends, or sub-topics relevant to the query]
{summaries_mid_level_communities}
</Mid_Level_Context>

<Entity_Nodes>
[Use this to define key actors, terms, or direct relationships mentioned in the query]
{nodes_context}
</Entity_Nodes>

<Specific_Documents>
[Use this for exact quotes, granular facts, and highly specific details]
{regular_rag_documents_texts}
</Specific_Documents>

<Visual_Context>
[If images are attached, incorporate their contents into your reasoning]
(Images attached in multimodal payload)
</Visual_Context>

---
USER QUERY:
{user_query}
---
"""

GRAPH_SUMMARY_SYSTEM_PROMPT = """
You are an expert in knowledge synthesis and technical summarization.

Your task is to transform structured knowledge (entities + relationships) into a dense, high-signal summary optimized for retrieval in a RAG system.

### OBJECTIVE:
Generate a compact, information-rich representation that:
- Preserves all critical technical facts and metrics
- Consolidates duplicate or conflicting values (prefer most consistent or repeated signals)
- Removes redundancy and noise
- Filters out invalid, weak, or illogical relationships
- Clearly separates model structure, inputs/outputs, and performance

### INPUT:
You will receive:
1. ENTITIES: Named concepts with descriptions
2. RELATIONSHIPS: Triplets connecting entities

### PROCESSING RULES:
- Deduplicate metrics (e.g., multiple F1/accuracy values -> summarize as ranges or most representative values)
- Ignore contradictory or after reasoning low-confidence relationships unless strongly supported
- Normalize synonyms (e.g., "optimal shade", "optimal threshold", "cutoff value" -> one concept)

### OUTPUT FORMAT:
1. Dense Summary (5-8 sentences max)
- Highly compressed, technical narrative
- Must include: model types, inputs, outputs, key metrics, thresholds, and interpretability approach

2. Structured Key Insights (bullet points)
- Max 8 bullets
- Each bullet = one atomic, high-value insight
- Prefer normalized terminology and grouped metrics

### STYLE:
- Technical and precise
- High signal-to-noise ratio
- No repetition
- No explanations or commentary
- No hallucinated connections

### CONSTRAINT:
Only use information grounded in the provided entities and relationships.
Do not infer beyond the data unless necessary for normalization.
Return only the final summary and bullet points.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive multimodal GraphRAG CLI")

    parser.add_argument(
        "--data_base",
        type=str,
        default="./localdb",
        help="Folder for the regular multimodal retriever DB.",
    )
    parser.add_argument(
        "--node_db",
        type=str,
        default="./localdb/node_db",
        help="Folder for node retriever DB.",
    )
    parser.add_argument(
        "--mid_communities_db",
        type=str,
        default="./localdb/mid_communities",
        help="Folder for mid communities retriever DB.",
    )
    parser.add_argument(
        "--global_communities_db",
        type=str,
        default="./localdb/global_communities",
        help="Folder for global communities retriever DB.",
    )

    parser.add_argument(
        "--embedding_provider",
        type=str,
        default="ollama",
        help="Embedding provider used to load retrievers.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="embeddinggemma:latest",
        help="Embedding model used to load retrievers.",
    )

    parser.add_argument(
        "--rag_provider",
        type=str,
        default=None,
        help="Answer-generation LLM provider override.",
    )
    parser.add_argument(
        "--rag_model",
        type=str,
        default=None,
        help="Answer-generation LLM model override.",
    )

    parser.add_argument(
        "--graph_summary_provider",
        type=str,
        default=None,
        help="Graph-context summarization LLM provider override.",
    )
    parser.add_argument(
        "--graph_summary_model",
        type=str,
        default=None,
        help="Graph-context summarization LLM model override.",
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
        "--top_k",
        type=int,
        default=None,
        help="Optional top-k override for all retrievers.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run one query and exit. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--show_context",
        action="store_true",
        help="Print retrieved context sections before the final answer.",
    )

    return parser.parse_args()


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


def _resolve_model_config(
    provider,
    model_name,
    provider_env: str,
    model_env: str,
    default_provider: str,
    default_model: str,
    step_name: str,
):
    project_root = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=project_root / ".env")

    resolved_provider = (provider or os.getenv(provider_env) or default_provider).lower()
    resolved_model_name = model_name or os.getenv(model_env) or default_model
    llm_kwargs = _resolve_provider_api_kwargs(resolved_provider, step_name)

    return resolved_provider, resolved_model_name, llm_kwargs


def _resolve_neo4j_auth(args: argparse.Namespace) -> tuple[str, str]:
    password = args.neo4j_password or os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError(
            "Missing Neo4j password. Set --neo4j_password or NEO4J_PASSWORD env var."
        )
    return args.neo4j_user, password


def _invoke_model(model_name: str, provider: str, messages, **llm_kwargs) -> str:
    llm = LLMModel(
        model_name=model_name,
        provider=provider,
        temperature=0,
        **llm_kwargs,
    ).as_langchain_llm()
    chain = llm | StrOutputParser()
    return chain.invoke(messages)


def _decode_json_payload(payload):
    if isinstance(payload, bytes):
        raw = payload.decode("utf-8")
    else:
        raw = str(payload)
    return json.loads(raw)


def parse_communities(communities_retrieved):
    parsed = []
    for community in communities_retrieved:
        try:
            parsed.append(_decode_json_payload(community))
        except Exception:
            continue

    summaries = []
    for community in parsed:
        summary = community.get("summary")
        if summary:
            summaries.append(summary)

    if not summaries:
        return "No community summaries retrieved."

    return "\n----\n".join(summaries)


def parse_nodes(nodes_retrieved):
    nodes_parsed = []
    for node in nodes_retrieved:
        try:
            parsed = _decode_json_payload(node)
        except Exception:
            continue

        if "name" not in parsed or "type" not in parsed:
            continue

        nodes_parsed.append(
            {
                "name": parsed.get("name", ""),
                "type": parsed.get("type", ""),
                "description": parsed.get("description", ""),
            }
        )
    return nodes_parsed


def get_graph_context(driver, database: str, retrieved_nodes):
    if not retrieved_nodes:
        return ""

    connections = {}
    entities = set()

    query = """
    MATCH (source)
    WHERE source.name = $name
      AND $label IN labels(source)
      AND coalesce(source.description, "") = $description

    MATCH path = (source)-[*1..2]-(neighbor)
    WHERE source <> neighbor
      AND NONE(lbl IN labels(neighbor) WHERE lbl IN ['text', 'image', 'table', 'document', 'file', 'code'])
      AND NONE(rel IN relationships(path) WHERE type(rel) = 'BELONGS_TO')

    WITH collect(path) AS paths
    UNWIND paths AS p
    UNWIND relationships(p) AS rel
    RETURN DISTINCT collect({
        subject: startNode(rel).name,
        subject_description: startNode(rel).description,
        predicate: type(rel),
        object: endNode(rel).name,
        object_description: endNode(rel).description
    }) AS connection_paths
    """

    with driver.session(database=database) as session:
        for node in retrieved_nodes:
            params = {
                "name": node["name"],
                "label": node["type"],
                "description": str(node.get("description") or ""),
            }

            results = session.run(query, **params)
            for record in results:
                for connection in record["connection_paths"]:
                    subject = connection["subject"]
                    subject_desc = connection["subject_description"] or ""
                    predicate = connection["predicate"]
                    object_ = connection["object"]
                    object_desc = connection["object_description"] or ""

                    entities.add((subject, subject_desc))
                    entities.add((object_, object_desc))

                    source_key = (subject, subject_desc)
                    if source_key not in connections:
                        connections[source_key] = {}
                    if predicate not in connections[source_key]:
                        connections[source_key][predicate] = set()

                    connections[source_key][predicate].add((object_, object_desc))

    if not entities and not connections:
        return ""

    entities_narrative = [
        f"\t'{entity_name}': {entity_desc or 'No description.'}"
        for entity_name, entity_desc in sorted(entities)
    ]

    connections_narrative = []
    for (subject, _subject_desc), predicate_to_objects in sorted(connections.items()):
        for predicate, object_pairs in sorted(predicate_to_objects.items()):
            formatted_predicate = predicate.replace("_", " ").lower()
            for object_name, _object_desc in sorted(object_pairs):
                connections_narrative.append(
                    f"\t'{subject}' {formatted_predicate} '{object_name}'"
                )

    summary_block = (
        "ENTITIES:\n"
        + "\n".join(entities_narrative)
        + "\n\nRELATIONSHIPS:\n"
        + "\n".join(connections_narrative)
    )

    return summary_block


def summarize_graph_context(nodes_context: str, model_name: str, provider: str, **llm_kwargs):
    if not nodes_context.strip():
        return "No entity context retrieved."

    messages = [
        SystemMessage(content=GRAPH_SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=nodes_context),
    ]
    return _invoke_model(model_name=model_name, provider=provider, messages=messages, **llm_kwargs)


def retrieve_context_for_nodes(
    nodes_retrieved,
    neo4j_uri: str,
    neo4j_auth: tuple[str, str],
    neo4j_database: str,
    model_name: str,
    provider: str,
    **llm_kwargs,
):
    nodes_parsed = parse_nodes(nodes_retrieved)
    if not nodes_parsed:
        return "No entity context retrieved."

    with GraphDatabase.driver(neo4j_uri, auth=neo4j_auth) as driver:
        nodes_context = get_graph_context(
            driver,
            database=neo4j_database,
            retrieved_nodes=nodes_parsed,
        )

    return summarize_graph_context(
        nodes_context,
        model_name=model_name,
        provider=provider,
        **llm_kwargs,
    )


def build_prompt(
    input_dict,
    system_prompt=RAG_SYSTEM_PROMPT,
    user_prompt=RAG_USER_PROMPT,
):
    global_context = input_dict.get(
        "summaries_global_level_communities", "No global context retrieved."
    )
    mid_context = input_dict.get(
        "summaries_mid_level_communities", "No mid-level context retrieved."
    )
    nodes_context = input_dict.get("nodes_context", "No entity context retrieved.")

    regular_texts_list = input_dict.get("regular_rag_documents", {}).get("texts", [])
    if regular_texts_list:
        formatted_regular_texts = "\n\n".join(
            [f"[Source {i + 1}]: {text}" for i, text in enumerate(regular_texts_list)]
        )
    else:
        formatted_regular_texts = "No specific documents retrieved."

    text_content = user_prompt.format(
        summaries_global_level_communities=global_context,
        summaries_mid_level_communities=mid_context,
        nodes_context=nodes_context,
        regular_rag_documents_texts=formatted_regular_texts,
        user_query=input_dict.get("user_query", ""),
    )

    user_content = [
        {
            "type": "text",
            "text": text_content,
        }
    ]

    images_list = input_dict.get("regular_rag_documents", {}).get("images", [])
    for img_b64 in images_list:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    return [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]


def _apply_top_k(retriever, top_k: int | None):
    if top_k is None:
        return retriever
    if top_k < 1:
        raise ValueError("--top_k must be >= 1")
    retriever.search_kwargs = {"k": top_k}
    return retriever


def _build_retrievers(args: argparse.Namespace):
    regular_rag_retriever = generate_database_and_retriever(
        main_folder=args.data_base,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )
    nodes_retriever = generate_database_and_retriever(
        main_folder=args.node_db,
        db_name="node_db",
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )
    mid_level_retriever = generate_database_and_retriever(
        main_folder=args.mid_communities_db,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )
    global_level_retriever = generate_database_and_retriever(
        main_folder=args.global_communities_db,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
    )

    return {
        "regular": _apply_top_k(regular_rag_retriever, args.top_k),
        "nodes": _apply_top_k(nodes_retriever, args.top_k),
        "mid": _apply_top_k(mid_level_retriever, args.top_k),
        "global": _apply_top_k(global_level_retriever, args.top_k),
    }


def _retrieve_all(retrievers: dict, question: str):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            key: executor.submit(retriever.invoke, question)
            for key, retriever in retrievers.items()
        }
        outputs = {key: future.result() for key, future in futures.items()}

    return outputs


def _answer_question(
    question: str,
    retrievers: dict,
    args: argparse.Namespace,
    rag_model_config,
    graph_model_config,
    neo4j_auth,
):
    rag_provider, rag_model_name, rag_kwargs = rag_model_config
    graph_provider, graph_model_name, graph_kwargs = graph_model_config

    retrieved = _retrieve_all(retrievers, question)

    regular_rag_documents = parse_documents(retrieved["regular"])
    summaries_mid_level_communities = parse_communities(retrieved["mid"])
    summaries_global_level_communities = parse_communities(retrieved["global"])
    nodes_context = retrieve_context_for_nodes(
        nodes_retrieved=retrieved["nodes"],
        neo4j_uri=args.neo4j_uri,
        neo4j_auth=neo4j_auth,
        neo4j_database=args.neo4j_database,
        model_name=graph_model_name,
        provider=graph_provider,
        **graph_kwargs,
    )

    prompt_payload = {
        "user_query": question,
        "regular_rag_documents": regular_rag_documents,
        "nodes_context": nodes_context,
        "summaries_mid_level_communities": summaries_mid_level_communities,
        "summaries_global_level_communities": summaries_global_level_communities,
    }
    messages = build_prompt(prompt_payload)

    if args.show_context:
        print("\n[Global Context]\n", summaries_global_level_communities)
        print("\n[Mid-Level Context]\n", summaries_mid_level_communities)
        print("\n[Entity Nodes]\n", nodes_context)
        print("\n[Specific Documents]\n", "\n".join(regular_rag_documents.get("texts", [])))

    result = _invoke_model(
        model_name=rag_model_name,
        provider=rag_provider,
        messages=messages,
        **rag_kwargs,
    )
    return result


def _interactive_loop(
    args: argparse.Namespace,
    retrievers: dict,
    rag_model_config,
    graph_model_config,
    neo4j_auth,
):
    print("Interactive GraphRAG CLI")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("rag> ").strip()
        except EOFError:
            print()
            return

        if not query:
            continue

        if query.lower() in {"exit", "quit", ":q"}:
            return

        try:
            answer = _answer_question(
                question=query,
                retrievers=retrievers,
                args=args,
                rag_model_config=rag_model_config,
                graph_model_config=graph_model_config,
                neo4j_auth=neo4j_auth,
            )
            print("\n" + answer + "\n")
        except Exception as exc:
            LOGGER.exception("Failed to answer query")
            print(f"Error: {exc}\n")


def _run_rag_cli() -> None:
    args = parse_args()

    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top_k must be >= 1")

    rag_model_config = _resolve_model_config(
        provider=args.rag_provider,
        model_name=args.rag_model,
        provider_env="RAG_LLM_PROVIDER",
        model_env="RAG_LLM_MODEL",
        default_provider="ollama",
        default_model="gemma3:12b",
        step_name="RAG generation",
    )
    graph_model_config = _resolve_model_config(
        provider=args.graph_summary_provider,
        model_name=args.graph_summary_model,
        provider_env="GRAPH_SUMMARY_LLM_PROVIDER",
        model_env="GRAPH_SUMMARY_LLM_MODEL",
        default_provider="ollama",
        default_model="gemma3:latest",
        step_name="graph summary",
    )
    neo4j_auth = _resolve_neo4j_auth(args)

    retrievers = _build_retrievers(args)

    if args.query:
        answer = _answer_question(
            question=args.query,
            retrievers=retrievers,
            args=args,
            rag_model_config=rag_model_config,
            graph_model_config=graph_model_config,
            neo4j_auth=neo4j_auth,
        )
        print(answer)
        return

    _interactive_loop(
        args=args,
        retrievers=retrievers,
        rag_model_config=rag_model_config,
        graph_model_config=graph_model_config,
        neo4j_auth=neo4j_auth,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(asctime)s %(message)s")
    _run_rag_cli()


if __name__ == "__main__":
    main()
