from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import logging

LOGGER = logging.getLogger(__name__)


class BatchTranslateResponse(BaseModel):
    translations: List[str] = Field(
        description="The list of translated strings in the exact same order as the input."
    )


def _translate_batch(texts: List[str], structured_llm):
    if not texts:
        return []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional translator. Translate the following list of strings into English. "
                "Maintain the exact order and return a JSON list.",
            ),
            ("human", "Translate these: {input_list}"),
        ]
    )

    chain = prompt | structured_llm.with_structured_output(BatchTranslateResponse)

    try:
        # We pass the whole list here
        result = chain.invoke({"input_list": texts})
        return result.translations
    except Exception as e:
        LOGGER.error(f"Batch translation failed: {e}")
        return texts  # Fallback to original if it fails


def translate_nodes(document_relationships, model_translate="gemma3:latest"):
    llm = ChatOllama(model=model_translate, temperature=0)

    # 1. Collect all unique strings that need translation
    to_translate = []
    for relationships in document_relationships.values():
        for relation in relationships:
            if getattr(relation, "language", "en") != "en":
                to_translate.append(relation.head)
                to_translate.append(relation.tail)

    if not to_translate:
        return document_relationships

    # 2. Process in batches of 15
    batch_size = 15
    translated_map = {}

    for i in range(0, len(to_translate), batch_size):
        current_batch = to_translate[i : i + batch_size]
        LOGGER.info(f"Translating batch {i // batch_size + 1}...")
        translated_results = _translate_batch(current_batch, llm)

        # Map original string -> translated string
        for original, translated in zip(current_batch, translated_results):
            translated_map[original] = translated

    # 3. Re-assign the translated values back to the objects
    for relationships in document_relationships.values():
        for relation in relationships:
            if getattr(relation, "language", "en") != "en":
                relation.head = translated_map.get(relation.head, relation.head)
                relation.tail = translated_map.get(relation.tail, relation.tail)
                relation.language = "en"  # Mark as done

    return document_relationships
