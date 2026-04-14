from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import logging
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from utils.models import LLMModel

LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()


class Relationship(BaseModel):
    head_id: Optional[str] = Field(
        default=None,
        description="Internal identifier for the head generated after extraction.",
    )
    head: str = Field(
        description="Normalized subject name. Use full names (e.g., 'Elon Musk' not 'Musk'). Translate to English."
    )
    head_type: str = Field(
        description="Upper-case category (e.g., PERSON, TECH, GEO, ORG)."
    )
    head_description: str = Field(description="Brief description of the subject.")
    head_language: str = Field(
        description="The language of 'head' field in ISO 3 letters code in lowercase."
    )
    relation: str = Field(
        description="Predicate in SCREAMING_SNAKE_CASE. Use specific verbs (e.g., ACQUIRED_BY instead of HAS)."
    )
    tail_id: Optional[str] = Field(
        default=None,
        description="Internal identifier for the tail generated after extraction.",
    )
    tail: str = Field(description="Normalized object name. Translate to English.")
    tail_type: str = Field(description="Upper-case category for the object.")
    tail_description: str = Field(description="Brief description of the object.")
    tail_language: str = Field(
        description="The language of 'tail' field in ISO 3 letters code in lowercase."
    )
    confidence: float = Field(
        description="Likelihood the relationship is explicitly supported by text (0.0-1.0)."
    )
    context: str = Field(
        description="A short quote from the text justifying this link."
    )


class KnowledgeGraph(BaseModel):
    """Encapsulates multiple relationships extracted from a text block."""

    relationships: List[Relationship]


def _resolve_llm(llm=None, model_name="gemma3:12b", provider="ollama", **llm_kwargs):
    if llm is not None:
        if hasattr(llm, "as_langchain_llm"):
            return llm.as_langchain_llm()
        return llm
    return LLMModel(
        model_name=model_name,
        provider=provider,
        **llm_kwargs,
    ).as_langchain_llm()


def _get_thread_chains(
    model_name_text,
    model_name_table,
    model_name_image,
    model_name_code,
    provider_text,
    provider_table,
    provider_image,
    provider_code,
    temperature,
):
    key = (
        model_name_text,
        model_name_table,
        model_name_image,
        model_name_code,
        provider_text,
        provider_table,
        provider_image,
        provider_code,
        temperature,
    )
    if getattr(_THREAD_LOCAL, "kg_chain_key", None) != key:
        _THREAD_LOCAL.chain_text = chain_for_text(
            model_name=model_name_text,
            provider=provider_text,
            temperature=temperature,
        )
        _THREAD_LOCAL.chain_table = chain_for_tables(
            model_name=model_name_table,
            provider=provider_table,
            temperature=temperature,
        )
        _THREAD_LOCAL.chain_image = chain_for_images(
            model_name=model_name_image,
            provider=provider_image,
            temperature=temperature,
        )
        _THREAD_LOCAL.chain_code = chain_for_code(
            model_name=model_name_code,
            provider=provider_code,
            temperature=temperature,
        )
        _THREAD_LOCAL.kg_chain_key = key
    return (
        _THREAD_LOCAL.chain_text,
        _THREAD_LOCAL.chain_table,
        _THREAD_LOCAL.chain_image,
        _THREAD_LOCAL.chain_code,
    )


def chain_for_text(model_name="gemma3:12b", provider="ollama", temperature=0, llm=None):
    llm = _resolve_llm(
        llm=llm,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
    )
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert Ontologist and Knowledge Graph Engineer. Your goal is to convert "
                    "unstructured text into a structured, machine-readable graph of atomic facts.\n\n"
                    "GUIDELINES:\n"
                    "1. **Entity Normalization**: Always use the most complete version of a name found in the text. "
                    "Resolve pronouns (he, she, it, they) to their original nouns.\n"
                    "2. **Relationship Granularity**: Prefer specific predicates over generic ones. "
                    "Use 'CHIEF_EXECUTIVE_OFFER_OF' instead of 'WORKS_AT'.\n"
                    "3. **Atomicity**: Each relationship should represent a single triplet. If a sentence "
                    "contains multiple facts, break them into multiple relationships.\n"
                    "4. **Translation**: Regardless of the input language, all extracted entities and "
                    "relations must be in English.\n"
                    "5. **Strictness**: Only extract information explicitly stated or strongly implied. "
                    "Do not hallucinate external knowledge about the entities.\n\n"
                    "{format_instructions}"
                ),
            ),
            ("human", "Input Text: {input}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def chain_for_tables(
    model_name="gemma3:12b",
    provider="ollama",
    temperature=0,
    llm=None,
):
    llm = _resolve_llm(
        llm=llm,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
    )
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)

    system_instruction = (
        "You are an expert Data Engineer specializing in Tabular Knowledge Extraction. "
        "Your goal is to transform Markdown, CSV, or HTML tables into a structured knowledge graph.\n\n"
        "SPECIFIC TABLE RULES:\n"
        "1. **Row-to-Triplet Mapping**: Generally, the primary entity in the first column is the 'Head'. "
        "The column headers represent the 'Relation', and the cell values are the 'Tail'.\n"
        "2. **Context Preservation**: If the table has a caption or a title, use it to normalize the entities "
        "(e.g., if the table is '2023 Financials', the year 2023 should be part of the relationship or entity).\n"
        "3. **Handle Empty Cells**: Do not create triplets for null, N/A, or empty cells.\n"
        "4. **Unit Integration**: Always include units (e.g., USD, kg, %) within the 'Tail' name or 'Relation' "
        "to ensure the data is meaningful.\n"
        "5. **Predicate Formatting**: Convert column headers into SCREAMING_SNAKE_CASE predicates "
        "(e.g., 'Revenue (Q1)' becomes 'REVENUE_Q1').\n\n"
        "{format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            ("human", "Background Context for the following table: {context}"),
            ("human", "Table Description: {description}"),
            (
                "human",
                "Extract all relationships from the following table data. "
                "Ensure numerical values are associated with their specific headers.\n\n"
                "Table Input:\n{input}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def chain_for_images(
    model_name="gemma3:12b",
    provider="ollama",
    temperature=0,
    llm=None,
):
    # Ensure the model used supports vision/multimodality
    llm = _resolve_llm(
        llm=llm,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
    )
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)

    system_instruction = (
        "You are an expert Knowledge Graph Engineer and Technical Systems Analyst. "
        "Your task is to decompose the provided technical image into a structured knowledge graph.\n\n"
        "TECHNICAL IMAGE ANALYSIS RULES:\n"
        "1. **Entity Identification**: Identify all nodes, blocks, icons, or text elements as entities. "
        "Include abstract concepts (e.g., 'Neural Network') and specific components (e.g., 'ReLU Layer'). "
        "Use precise nomenclature found in the image.\n"
        "2. **Functional Relationships**: Identify connections between entities. Instead of physical "
        "space, focus on logical flow. Use predicates like 'INPUT_TO', 'DEPENDS_ON', 'REFINES', "
        "'INHERITS_FROM', or 'STORES'.\n"
        "3. **Directionality & Flow**: Treat arrows, lines, and connectors as directed edges. "
        "The 'Head' is the source and the 'Tail' is the destination of the logical flow.\n"
        "4. **Categorical Hierarchy**: Use 'PART_OF' or 'INSTANCE_OF' to represent elements contained "
        "within a boundary box or grouping.\n"
        "5. **Technical Attributes**: Map visual styling to metadata (e.g., Head: 'Database', "
        "Relation: 'IS_ENCRYPTED', Tail: 'True') if represented by specific colors or dashed lines.\n"
        "6. **Diagram Context**: Classify the diagram type (e.g., 'UML Diagram', 'Cloud Architecture', "
        "'Decision Tree') and relate the root entities to this context.\n\n"
        "{format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            ("human", "Background Context for the following image: {context}"),
            ("human", "Image Description: {description}"),
            (
                "human",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{input}"},
                    },
                ],
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def chain_for_code(model_name="gemma3:12b", provider="ollama", temperature=0, llm=None):
    llm = _resolve_llm(
        llm=llm,
        model_name=model_name,
        provider=provider,
        temperature=temperature,
    )
    parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)

    system_instruction = (
        "You are an expert software architect and Knowledge Graph Engineer. "
        "Your task is to extract explicit, code-grounded relationships from source code.\n\n"
        "CODE EXTRACTION RULES:\n"
        "1. Prefer concrete software entities such as classes, functions, modules, files, APIs, databases, and services.\n"
        "2. Use specific predicates in SCREAMING_SNAKE_CASE such as CALLS, IMPORTS, INHERITS_FROM, IMPLEMENTS, RETURNS, WRITES_TO.\n"
        "3. Extract only relationships supported by the provided code or context; do not infer external behavior.\n"
        "4. Normalize names to stable identifiers when possible (e.g., module.Class.method).\n"
        "5. Keep relationships atomic and preserve directional flow.\n\n"
        "{format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            ("human", "Background Context for the following code: {context}"),
            ("human", "Code Description: {description}"),
            ("human", "Code Input:\n{input}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


async def aextract_relationships_from_element(
    element,
    element_id,
    model_name_text="gemma3:12b",
    model_name_table="gemma3:12b",
    model_name_image="gemma3:12b",
    model_name_code="gemma3:12b",
    provider_text="ollama",
    provider_table="ollama",
    provider_image="ollama",
    provider_code="ollama",
    temperature=0,
    chain_text=None,
    chain_table=None,
    chain_image=None,
    chain_code=None,
):

    type_ = element.get("type", None)
    content = element.get("content", None)
    description = element.get("description", None)
    context = element.get("context", None)
    if content is None:
        LOGGER.warning("No content found for element: {}".format(element_id))
        return

    if type_ is None:
        LOGGER.warning("No type found for element: {}".format(element_id))
        return None

    try:
        if type_ == "table":
            LOGGER.info(
                "Extracting relationships from table element: {}".format(element_id)
            )
            chain = chain_table or chain_for_tables(
                model_name=model_name_table,
                provider=provider_table,
                temperature=temperature,
            )

            response = await chain.ainvoke(
                {"input": content, "description": description, "context": context}
            )

            LOGGER.info(
                "Extracting relationships from description of table element: {}".format(
                    element_id
                )
            )

            return response.relationships

        elif type_ == "image":
            LOGGER.info(
                "Extracting relationships from image element: {}".format(element_id)
            )
            chain = chain_image or chain_for_images(
                model_name=model_name_image,
                provider=provider_image,
                temperature=temperature,
            )
            response = await chain.ainvoke(
                {"input": content, "description": description, "context": context}
            )
            return response.relationships

        elif type_ == "text":
            LOGGER.info("Extracting relationships from text element: {}".format(element_id))
            chain = chain_text or chain_for_text(
                model_name=model_name_text,
                provider=provider_text,
                temperature=temperature,
            )
            response = await chain.ainvoke({"input": content})
            return response.relationships

        elif type_ in {"code", "notebook_code", "python"}:
            LOGGER.info("Extracting relationships from code element: {}".format(element_id))
            chain = chain_code or chain_for_code(
                model_name=model_name_code,
                provider=provider_code,
                temperature=temperature,
            )
            response = await chain.ainvoke(
                {"input": content, "description": description, "context": context}
            )
            return response.relationships

        else:
            LOGGER.warning("Unknown element type: {}".format(type_))
            return []
    except Exception as exc:
        LOGGER.warning(
            "KG extraction failed for element=%s type=%s. Skipping element. Error: %s",
            element_id,
            type_,
            exc,
        )
        return []


def parse_document_to_dict(doc):
    if isinstance(doc, dict):
        data = dict(doc)
    else:
        decoded_document = doc.decode("utf-8") if isinstance(doc, bytes) else str(doc)
        data = json.loads(decoded_document)

    if "metadata" in data and isinstance(data["metadata"], str):
        data["metadata"] = json.loads(data["metadata"])
    return data


def extract_relationships_from_element(
    element,
    element_id,
    chain_text,
    chain_table,
    chain_image,
    chain_code,
):
    type_ = element.get("type", None)
    content = element.get("content", None)
    description = element.get("description", None)
    context = element.get("context", None)

    if content is None:
        LOGGER.warning("No content found for element: %s", element_id)
        return None
    if type_ is None:
        LOGGER.warning("No type found for element: %s", element_id)
        return None

    try:
        if type_ == "table":
            response = chain_table.invoke(
                {"input": content, "description": description, "context": context}
            )
            return response.relationships
        if type_ == "image":
            response = chain_image.invoke(
                {"input": content, "description": description, "context": context}
            )
            return response.relationships
        if type_ == "text":
            response = chain_text.invoke({"input": content})
            return response.relationships
        if type_ in {"code", "notebook_code", "python"}:
            response = chain_code.invoke(
                {"input": content, "description": description, "context": context}
            )
            return response.relationships
    except Exception as exc:
        LOGGER.warning(
            "KG extraction failed for element=%s type=%s. Skipping element. Error: %s",
            element_id,
            type_,
            exc,
        )
        return []

    LOGGER.warning("Unknown element type: %s", type_)
    return []


def _thread_worker(
    doc_id,
    element_index,
    element,
    model_name_text,
    model_name_table,
    model_name_image,
    model_name_code,
    provider_text,
    provider_table,
    provider_image,
    provider_code,
    temperature,
):
    chain_text, chain_table, chain_image, chain_code = _get_thread_chains(
        model_name_text=model_name_text,
        model_name_table=model_name_table,
        model_name_image=model_name_image,
        model_name_code=model_name_code,
        provider_text=provider_text,
        provider_table=provider_table,
        provider_image=provider_image,
        provider_code=provider_code,
        temperature=temperature,
    )
    relationships = extract_relationships_from_element(
        element=element,
        element_id=f"{doc_id}::element::{element_index}",
        chain_text=chain_text,
        chain_table=chain_table,
        chain_image=chain_image,
        chain_code=chain_code,
    )
    return doc_id, element_index, relationships


def _expand_documents_to_elements(
    documents_dict,
) -> List[Tuple[str, int, Dict[str, Any]]]:
    expanded: List[Tuple[str, int, Dict[str, Any]]] = []

    for doc_id, doc in documents_dict.items():
        parsed_doc = parse_document_to_dict(doc)
        if not isinstance(parsed_doc, dict):
            LOGGER.warning("Skipping doc %s: parsed payload is not a dict.", doc_id)
            continue

        parsed_elements = parsed_doc.get("elements")
        if isinstance(parsed_elements, list) and parsed_elements:
            added_any = False
            for element_index, element in enumerate(parsed_elements):
                if not isinstance(element, dict):
                    LOGGER.warning(
                        "Skipping non-dict element in doc %s at index %d.",
                        doc_id,
                        element_index,
                    )
                    continue
                expanded.append((doc_id, element_index, element))
                added_any = True
            if added_any:
                continue

        expanded.append((doc_id, 0, parsed_doc))

    return expanded


def convert_to_graph_elements_pipeline_threaded(
    documents_dict,
    model_name_text="gemma3:12b",
    model_name_table="gemma3:12b",
    model_name_image="gemma3:12b",
    model_name_code="gemma3:12b",
    provider_text="ollama",
    provider_table="ollama",
    provider_image="ollama",
    provider_code="ollama",
    temperature=0,
    max_workers=5,
):
    expanded_elements = _expand_documents_to_elements(documents_dict)
    grouped_results: Dict[str, List[Tuple[int, List[Relationship]]]] = defaultdict(list)

    if not expanded_elements:
        return {}

    LOGGER.info(
        "KG extraction (thread mode): %d documents expanded to %d elements using %d workers.",
        len(documents_dict),
        len(expanded_elements),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _thread_worker,
                doc_id,
                element_index,
                element,
                model_name_text,
                model_name_table,
                model_name_image,
                model_name_code,
                provider_text,
                provider_table,
                provider_image,
                provider_code,
                temperature,
            )
            for doc_id, element_index, element in expanded_elements
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            doc_id, element_index, relationships = future.result()
            if relationships is not None:
                grouped_results[doc_id].append((element_index, relationships))

    results: Dict[str, List[Relationship]] = {}
    for doc_id, indexed_relationships in grouped_results.items():
        merged_relationships: List[Relationship] = []
        for _, relationships in sorted(indexed_relationships, key=lambda item: item[0]):
            merged_relationships.extend(relationships)
        results[doc_id] = merged_relationships

    return results


async def convert_to_graph_elements_pipeline(
    documents_dict,
    model_name_text="gemma3:12b",
    model_name_table="gemma3:12b",
    model_name_image="gemma3:12b",
    model_name_code="gemma3:12b",
    provider_text="ollama",
    provider_table="ollama",
    provider_image="ollama",
    provider_code="ollama",
    temperature=0,
    max_concurrency=5,
    execution_mode="async",
):
    if execution_mode == "thread":
        return await asyncio.to_thread(
            convert_to_graph_elements_pipeline_threaded,
            documents_dict,
            model_name_text,
            model_name_table,
            model_name_image,
            model_name_code,
            provider_text,
            provider_table,
            provider_image,
            provider_code,
            temperature,
            max_concurrency,
        )

    expanded_elements = _expand_documents_to_elements(documents_dict)
    if not expanded_elements:
        return {}

    LOGGER.info(
        "KG extraction (async mode): %d documents expanded to %d elements with max_concurrency=%d.",
        len(documents_dict),
        len(expanded_elements),
        max_concurrency,
    )

    chain_text = chain_for_text(
        model_name=model_name_text,
        provider=provider_text,
        temperature=temperature,
    )
    chain_table = chain_for_tables(
        model_name=model_name_table,
        provider=provider_table,
        temperature=temperature,
    )
    chain_image = chain_for_images(
        model_name=model_name_image,
        provider=provider_image,
        temperature=temperature,
    )
    chain_code = chain_for_code(
        model_name=model_name_code,
        provider=provider_code,
        temperature=temperature,
    )

    sem = asyncio.Semaphore(max_concurrency)

    async def throttled_extraction(doc_id, element_index, element):
        async with sem:
            relationship = await aextract_relationships_from_element(
                element,
                f"{doc_id}::element::{element_index}",
                model_name_text,
                model_name_table,
                model_name_image,
                model_name_code,
                provider_text,
                provider_table,
                provider_image,
                provider_code,
                temperature,
                chain_text,
                chain_table,
                chain_image,
                chain_code,
            )
            return doc_id, element_index, relationship

    # Create tasks using the throttled wrapper
    tasks = [
        throttled_extraction(doc_id, element_index, element)
        for doc_id, element_index, element in expanded_elements
    ]

    # Gather results
    results = await tqdm_asyncio.gather(*tasks)

    grouped_results: Dict[str, List[Tuple[int, List[Relationship]]]] = defaultdict(list)
    for doc_id, element_index, relationships in results:
        if relationships is None:
            continue
        grouped_results[doc_id].append((element_index, relationships))

    merged_by_document: Dict[str, List[Relationship]] = {}
    for doc_id, indexed_relationships in grouped_results.items():
        merged_relationships: List[Relationship] = []
        for _, relationships in sorted(indexed_relationships, key=lambda item: item[0]):
            merged_relationships.extend(relationships)
        merged_by_document[doc_id] = merged_relationships

    return merged_by_document


def fetch_existing_neo4j_nodes(
    uri: str,
    auth: tuple[str, str],
    database: str = "neo4j",
) -> List[Dict[str, Any]]:
    """Fetch existing graph nodes from Neo4j as lightweight records."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "Neo4j driver is not available. Install dependency `neo4j`."
        ) from exc

    query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
    RETURN n.name AS name, n.description AS description, labels(n) AS labels
    """

    nodes: List[Dict[str, Any]] = []
    with GraphDatabase.driver(uri, auth=auth) as driver:
        with driver.session(database=database) as session:
            records = session.run(query)
            for record in records:
                labels = record.get("labels") or []
                name = record.get("name")
                if not name or not labels:
                    continue
                nodes.append(
                    {
                        "name": str(name),
                        "type": str(labels[0]),
                        "description": str(record.get("description") or ""),
                    }
                )
    return nodes
