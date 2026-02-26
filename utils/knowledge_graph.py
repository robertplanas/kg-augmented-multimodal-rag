from pyexpat import model
from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import logging
import asyncio
import json
from tqdm.asyncio import tqdm_asyncio

LOGGER = logging.getLogger(__name__)


class Relationship(BaseModel):
    head: str = Field(
        description="Normalized subject name. Use full names (e.g., 'Elon Musk' not 'Musk'). Translate to English."
    )
    head_type: str = Field(
        description="Upper-case category (e.g., PERSON, TECH, GEO, ORG)."
    )
    relation: str = Field(
        description="Predicate in SCREAMING_SNAKE_CASE. Use specific verbs (e.g., ACQUIRED_BY instead of HAS)."
    )
    tail: str = Field(description="Normalized object name. Translate to English.")
    tail_type: str = Field(description="Upper-case category for the object.")
    confidence: float = Field(
        description="Likelihood the relationship is explicitly supported by text (0.0-1.0)."
    )
    context: str = Field(
        description="A short quote from the text justifying this link."
    )
    language: str = Field(
        description="The language of the text in ISO 3 letters code in lowercase."
    )


class KnowledgeGraph(BaseModel):
    """Encapsulates multiple relationships extracted from a text block."""

    relationships: List[Relationship]


def chain_for_text(model_name="gemma3:12b", temperature=0):

    llm = ChatOllama(temperature=temperature, model=model_name)
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


def chain_for_tables(model_name="gemma3:12b", temperature=0):
    llm = ChatOllama(temperature=temperature, model=model_name)
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


def chain_for_images(model_name="gemma3:12b", temperature=0):
    # Ensure the model used supports vision/multimodality
    llm = ChatOllama(temperature=temperature, model=model_name)
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


async def aextract_relationships_from_element(
    element,
    element_id,
    model_name_text="gemma3:12b",
    model_name_table="gemma3:12b",
    model_name_image="gemma3:12b",
    temperature=0,
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

    if type_ == "table":
        LOGGER.info(
            "Extracting relationships from table element: {}".format(element_id)
        )
        chain = chain_for_tables(model_name=model_name_table, temperature=temperature)

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
        chain = chain_for_images(model_name=model_name_image, temperature=temperature)
        response = await chain.ainvoke(
            {"input": content, "description": description, "context": context}
        )
        return response.relationships

    elif type_ == "text":
        LOGGER.info("Extracting relationships from text element: {}".format(element_id))
        chain = chain_for_text(model_name=model_name_text, temperature=temperature)
        response = await chain.ainvoke({"input": content})
        return response.relationships

    else:
        LOGGER.warning("Unknown element type: {}".format(type_))
        return None


def parse_document_to_dict(doc):

    decoded_document = doc.decode("utf-8")
    data = json.loads(decoded_document)

    if "metadata" in data:
        data["metadata"] = json.loads(data["metadata"])
    return data


async def convert_to_graph_elements_pipeline(
    documents_dict,
    model_name_text="gemma3:12b",
    model_name_table="gemma3:12b",
    model_name_image="gemma3:12b",
    temperature=0,
    max_concurrency=5,
):

    #### PEDNING ######

    # I would like to implement the option of selecting another chat like OpenAI ChatGPT or Gemini
    # to be used in the pipeline

    ##################

    sem = asyncio.Semaphore(max_concurrency)

    async def throttled_extraction(doc_id, doc):
        async with sem:
            # This task "holds" a spot until it returns

            doc = parse_document_to_dict(doc)

            return await aextract_relationships_from_element(
                doc,
                doc_id,
                model_name_text,
                model_name_table,
                model_name_image,
                temperature,
            )

    relationship_elements = []

    # Create tasks using the throttled wrapper
    tasks = [
        throttled_extraction(doc_id, doc) for doc_id, doc in documents_dict.items()
    ]

    # Gather results
    relationship_elements = await tqdm_asyncio.gather(*tasks)

    # Filter out None values
    return [el for el in relationship_elements if el is not None]
