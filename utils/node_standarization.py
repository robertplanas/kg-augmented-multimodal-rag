from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import logging
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

LOGGER = logging.getLogger(__name__)


UNIVERSAL_TO_WORDNET = {
    "ADJ": wordnet.ADJ,
    "ADV": wordnet.ADV,
    "NOUN": wordnet.NOUN,
    "PRON": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "AUX": wordnet.VERB,
}


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


def get_wordnet_pos_universal(tag):
    """
    Returns the WordNet constant for a given Universal Dependency tag.
    Defaults to wordnet.NOUN for tags that don't have a specific lemma form.
    """
    return UNIVERSAL_TO_WORDNET.get(tag, wordnet.NOUN)


def get_wordnet_pos(sentence):
    """Map POS tag to first character lemmatize() accepts"""
    tags = nltk.pos_tag(word_tokenize(sentence), tagset="universal")


def lemmatize_text(node, lemmatizer):
    """Lemmatizes a string (handles multi-word phrases by lemmatizing each word)"""
    if not node:
        return node

    words_and_tags = nltk.pos_tag(word_tokenize(node), tagset="universal")
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos_universal(tag))
        for word, tag in words_and_tags
    ]
    return " ".join(lemmatized_words)


def standarize_format_lematizer(original_val, field):

    if field in ["head", "tail"]:
        return original_val.lower()
    elif field in ["tail_type", "head_type"]:
        return original_val.replace("_", " ").lower()
    else:
        raise ValueError(f"Unknown field: {field}")


def standarize_format(original_val, field):

    if field in ["head", "tail"]:
        return original_val.lower()
    elif field in ["tail_type", "head_type"]:
        return original_val.lower().replace(" ", "_").lower()
    else:
        raise ValueError(f"Unknown field: {field}")


def lemmatize_nodes_and_relationships(
    document_relationships,
):
    """
    Standardizes 'head', 'tail', and 'relation' strings using NLTK WordNetLemmatizer.
    This version is significantly faster than LLM-based lemmatization.
    """
    LOGGER.info("Starting NLTK-based lemmatization.")
    lemmatizer = WordNetLemmatizer()
    lemma_cache = {}
    for relationships in document_relationships.values():
        for relation in relationships:
            # Fields to process
            fields = ["head", "head_type", "tail", "tail_type"]
            for field in fields:
                original_val = getattr(relation, field)
                original_val = standarize_format_lematizer(original_val, field)

                if original_val not in lemma_cache:
                    lemmatized_val = lemmatize_text(original_val, lemmatizer)
                    lemmatized_val = standarize_format(lemmatized_val, field)
                    lemma_cache[original_val] = lemmatized_val

                setattr(relation, field, lemma_cache[original_val])

    return document_relationships
