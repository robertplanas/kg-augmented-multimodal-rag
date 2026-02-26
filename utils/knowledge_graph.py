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
        "You are an expert Computer Vision Engineer and Scene Graph Generator. "
        "Your task is to decompose the provided image into a structured knowledge graph.\n\n"
        "IMAGE ANALYSIS RULES:\n"
        "1. **Object Detection**: Every distinct object or person should be a 'Head' or 'Tail'. "
        "Use specific labels (e.g., 'Golden Retriever' instead of 'Dog').\n"
        "2. **Spatial Relationships**: Capture the physical layout using predicates like "
        "LEFT_OF, BEHIND, MOUNTED_ON, or INSIDE.\n"
        "3. **Action & Interaction**: Identify verbs between entities, such as "
        "WEARING, CARRYING, EATING, or LOOKING_AT.\n"
        "4. **Attribute Attribution**: Treat significant visual properties as relationships "
        "(e.g., Head: 'Shirt', Relation: 'HAS_COLOR', Tail: 'Crimson').\n"
        "5. **Scene Context**: If the image is a specific setting (e.g., a Kitchen), "
        "relate the main objects to that setting.\n\n"
        "{format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
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

        response = await chain.ainvoke({"input": content})

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
        response = await chain.ainvoke({"input": content})
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
