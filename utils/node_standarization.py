from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.prompts import ChatPromptTemplate
import logging
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nltk
from dotenv import load_dotenv
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, Field
from scipy.spatial.distance import cosine

from utils.models import EmbeddingModel, LLMModel

LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()


UNIVERSAL_TO_WORDNET = {
    "ADJ": wordnet.ADJ,
    "ADV": wordnet.ADV,
    "NOUN": wordnet.NOUN,
    "PRON": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "AUX": wordnet.VERB,
}

PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


class BatchTranslateResponse(BaseModel):
    translations: List[str] = Field(
        description="The list of translated strings in the exact same order as the input."
    )


class DescriptionSummaryResponse(BaseModel):
    summary: str = Field(
        description="A concise description that merges all provided descriptions into one."
    )


@dataclass
class ExtractedNode:
    name: str
    type: str
    description: str
    node_id: str
    embedding: Optional[List[float]] = None
    group_id: Optional[str] = None
    internal_id: Optional[str] = None


@dataclass
class RepresentativeNodeResult:
    group_id: str
    name: str
    type: str
    description: str
    mapping_node_ids: List[str]


def _resolve_translate_config(
    model_translate,
    provider,
    max_workers,
    batch_size,
    llm_kwargs,
):
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / ".env")
    load_dotenv(dotenv_path=project_root / ".venv" / ".env")

    resolved_provider = (
        provider
        or os.getenv("NODE_TRANSLATE_PROVIDER")
        or os.getenv("NODE_STANDARDIZATION_PROVIDER")
        or "ollama"
    ).lower()
    resolved_model = (
        model_translate
        or os.getenv("NODE_TRANSLATE_MODEL")
        or os.getenv("NODE_STANDARDIZATION_MODEL")
        or "gemma3:latest"
    )

    if max_workers is None:
        env_workers = os.getenv("NODE_TRANSLATE_MAX_WORKERS", "1")
        try:
            resolved_max_workers = int(env_workers)
        except ValueError:
            raise ValueError(
                f"Invalid NODE_TRANSLATE_MAX_WORKERS value: {env_workers}. Expected integer."
            ) from None
    else:
        resolved_max_workers = max_workers

    if resolved_max_workers < 1:
        raise ValueError("max_workers must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    api_key_env = PROVIDER_API_KEY_ENV.get(resolved_provider)
    if api_key_env and "api_key" not in llm_kwargs:
        api_key = os.getenv(api_key_env)
        if api_key:
            llm_kwargs["api_key"] = api_key
        else:
            raise ValueError(
                f"Missing API key for translation provider '{resolved_provider}'. "
                f"Set {api_key_env} in your environment or in {project_root / '.env'}."
            )

    return (
        resolved_model,
        resolved_provider,
        resolved_max_workers,
        batch_size,
        llm_kwargs,
    )


def _get_thread_llm(model_name, provider, llm_kwargs):
    key = (model_name, provider, tuple(sorted(llm_kwargs.items())))
    if getattr(_THREAD_LOCAL, "translate_llm_key", None) != key:
        _THREAD_LOCAL.translate_llm = LLMModel(
            model_name=model_name,
            provider=provider,
            temperature=0,
            **llm_kwargs,
        ).as_langchain_llm()
        _THREAD_LOCAL.translate_llm_key = key
    return _THREAD_LOCAL.translate_llm


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
        result = chain.invoke({"input_list": texts})
        translated = result.translations
        if len(translated) != len(texts):
            LOGGER.warning(
                "Translation output length mismatch. Falling back to original batch."
            )
            return texts
        return translated
    except Exception as exc:
        LOGGER.error("Batch translation failed: %s", exc)
        return texts


def _translate_batch_worker(
    batch_index,
    texts,
    model_translate,
    provider,
    llm_kwargs,
):
    llm = _get_thread_llm(model_translate, provider, llm_kwargs)
    translated = _translate_batch(texts, llm)
    return batch_index, translated


def translate_nodes(
    document_relationships,
    model_translate="gemma3:latest",
    provider=None,
    max_workers=None,
    batch_size=15,
    execution_mode="thread",
    **llm_kwargs,
):
    (
        model_translate,
        provider,
        max_workers,
        batch_size,
        llm_kwargs,
    ) = _resolve_translate_config(
        model_translate=model_translate,
        provider=provider,
        max_workers=max_workers,
        batch_size=batch_size,
        llm_kwargs=llm_kwargs,
    )

    to_translate = []
    for relationships in document_relationships.values():
        for relation in relationships:
            to_translate.append(relation.head)
            to_translate.append(relation.tail)

    if not to_translate:
        return document_relationships

    ordered_unique_to_translate = list(dict.fromkeys(to_translate))

    translated_map = {}
    batches = [
        ordered_unique_to_translate[i : i + batch_size]
        for i in range(0, len(ordered_unique_to_translate), batch_size)
    ]

    if execution_mode != "thread" or max_workers <= 1 or len(batches) <= 1:
        llm = LLMModel(
            model_name=model_translate,
            provider=provider,
            temperature=0,
            **llm_kwargs,
        ).as_langchain_llm()
        for batch_index, current_batch in enumerate(batches):
            LOGGER.info(
                "Translating batch %s/%s with provider=%s model=%s",
                batch_index + 1,
                len(batches),
                provider,
                model_translate,
            )
            translated_results = _translate_batch(current_batch, llm)
            for original, translated in zip(current_batch, translated_results):
                translated_map[original] = translated
    else:
        LOGGER.info(
            "Translating %d strings in %d batches using %d workers (provider=%s model=%s)",
            len(ordered_unique_to_translate),
            len(batches),
            max_workers,
            provider,
            model_translate,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _translate_batch_worker,
                    batch_index,
                    current_batch,
                    model_translate,
                    provider,
                    llm_kwargs,
                )
                for batch_index, current_batch in enumerate(batches)
            ]

            batch_results = []
            for future in as_completed(futures):
                batch_results.append(future.result())

        for batch_index, translated_results in sorted(
            batch_results, key=lambda x: x[0]
        ):
            current_batch = batches[batch_index]
            for original, translated in zip(current_batch, translated_results):
                translated_map[original] = translated

    for relationships in document_relationships.values():
        for relation in relationships:
            relation.head = translated_map.get(relation.head, relation.head)
            relation.head_language = "en"

            relation.tail = translated_map.get(relation.tail, relation.tail)
            relation.tail_language = "en"

    return document_relationships


def get_wordnet_pos_universal(tag):
    """
    Returns the WordNet constant for a given Universal Dependency tag.
    Defaults to wordnet.NOUN for tags that don't have a specific lemma form.
    """
    return UNIVERSAL_TO_WORDNET.get(tag, wordnet.NOUN)


def get_wordnet_pos(sentence):
    """Map POS tag to first character lemmatize() accepts"""
    nltk.pos_tag(word_tokenize(sentence), tagset="universal")


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
    if field in ["tail_type", "head_type"]:
        return original_val.replace("_", " ").lower()
    raise ValueError(f"Unknown field: {field}")


def standarize_format(original_val, field):

    if field in ["head", "tail"]:
        return original_val.lower()
    if field in ["tail_type", "head_type"]:
        return original_val.lower().replace(" ", "_").lower()
    raise ValueError(f"Unknown field: {field}")


def lemmatize_nodes_and_relationships(
    document_relationships,
):
    """
    Standardizes 'head', 'tail', and types using NLTK WordNetLemmatizer.
    """
    LOGGER.info("Starting NLTK-based lemmatization.")
    lemmatizer = WordNetLemmatizer()
    lemma_cache = {}
    for relationships in document_relationships.values():
        for relation in relationships:
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


class UnionFind:
    def __init__(self, ids: List[str]):
        self.parent = {node_id: node_id for node_id in ids}

    def find(self, node_id: str) -> str:
        if self.parent[node_id] == node_id:
            return node_id
        self.parent[node_id] = self.find(self.parent[node_id])
        return self.parent[node_id]

    def union(self, node_i: str, node_j: str) -> None:
        root_i = self.find(node_i)
        root_j = self.find(node_j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def build_nodes_from_relationships(
    document_relationships,
    embedding_model_name: str = "embeddinggemma:latest",
    embedding_provider: str = "ollama",
    **embedding_kwargs,
) -> List[ExtractedNode]:
    """Build node objects from relationship heads and tails, including embeddings."""
    embedding_model = EmbeddingModel(
        model_name=embedding_model_name,
        provider=embedding_provider,
        **embedding_kwargs,
    )

    nodes: List[ExtractedNode] = []
    embedding_cache: Dict[str, List[float]] = {}

    for document_id, relations in document_relationships.items():
        for index_relation, relationship in enumerate(relations):
            head_id = relationship.head_id or f"{document_id}_head_{index_relation}"
            tail_id = relationship.tail_id or f"{document_id}_tail_{index_relation}"
            relationship.head_id = head_id
            relationship.tail_id = tail_id

            head_name = str(relationship.head or "")
            tail_name = str(relationship.tail or "")

            if head_name not in embedding_cache:
                embedding_cache[head_name] = embedding_model.get_embedding(head_name)
            if tail_name not in embedding_cache:
                embedding_cache[tail_name] = embedding_model.get_embedding(tail_name)

            nodes.append(
                ExtractedNode(
                    name=head_name,
                    type=str(relationship.head_type or ""),
                    description=str(relationship.head_description or ""),
                    node_id=head_id,
                    embedding=embedding_cache[head_name],
                )
            )
            nodes.append(
                ExtractedNode(
                    name=tail_name,
                    type=str(relationship.tail_type or ""),
                    description=str(relationship.tail_description or ""),
                    node_id=tail_id,
                    embedding=embedding_cache[tail_name],
                )
            )

    return nodes


def group_nodes_by_similarity(
    nodes: List[ExtractedNode],
    threshold: float = 0.05,
) -> Dict[str, List[ExtractedNode]]:
    """Group nodes using pairwise cosine distance and union-find."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    if not nodes:
        return {}

    for index, node in enumerate(nodes):
        node.group_id = None
        node.internal_id = f"node_{index}"

    uf = UnionFind([n.internal_id for n in nodes if n.internal_id is not None])

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            if cosine(node_i.embedding or [], node_j.embedding or []) < threshold:
                uf.union(node_i.internal_id, node_j.internal_id)

    groups: Dict[str, List[ExtractedNode]] = defaultdict(list)
    for node in nodes:
        node.group_id = uf.find(node.internal_id)
        groups[node.group_id].append(node)

    return dict(groups)


def _summarize_descriptions(
    descriptions: List[str],
    model_name: str,
    provider: str,
    **llm_kwargs,
) -> str:
    if not descriptions:
        return ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Graph Data Analyst. Consolidate the provided node descriptions "
                "into one concise canonical description.",
            ),
            ("human", "Descriptions:\n{descriptions}"),
        ]
    )

    llm = LLMModel(
        model_name=model_name,
        provider=provider,
        temperature=0,
        **llm_kwargs,
    ).as_langchain_llm()
    chain = prompt | llm.with_structured_output(DescriptionSummaryResponse)

    try:
        response = chain.invoke({"descriptions": "\n".join(descriptions)})
        return response.summary
    except Exception as exc:
        LOGGER.warning(
            "Description summarization failed. Falling back to first value: %s", exc
        )
        return descriptions[0]


def _build_existing_nodes_with_embeddings(
    existing_nodes: List[Dict[str, Any]],
    embedding_model_name: str,
    embedding_provider: str,
    **embedding_kwargs,
) -> List[ExtractedNode]:
    if not existing_nodes:
        return []

    embedding_model = EmbeddingModel(
        model_name=embedding_model_name,
        provider=embedding_provider,
        **embedding_kwargs,
    )
    embedding_cache: Dict[str, List[float]] = {}

    parsed_nodes: List[ExtractedNode] = []
    for index, node in enumerate(existing_nodes):
        name = str(node.get("name") or "")
        if not name:
            continue
        if name not in embedding_cache:
            embedding_cache[name] = embedding_model.get_embedding(name)

        parsed_nodes.append(
            ExtractedNode(
                name=name,
                type=str(node.get("type") or "ENTITY"),
                description=str(node.get("description") or ""),
                node_id=f"existing_{index}",
                embedding=embedding_cache[name],
            )
        )

    return parsed_nodes


def match_group_to_existing_nodes(
    group_nodes: List[ExtractedNode],
    existing_nodes: List[ExtractedNode],
    threshold: float = 0.05,
) -> Optional[ExtractedNode]:
    if threshold < 0:
        raise ValueError("threshold must be >= 0")

    for node in group_nodes:
        for existing_node in existing_nodes:
            if cosine(node.embedding or [], existing_node.embedding or []) < threshold:
                return existing_node
    return None


def aggregate_group_metadata(
    group_nodes: List[ExtractedNode],
    description_model_name: str = "gemma3:12b",
    description_provider: str = "ollama",
    **description_llm_kwargs,
) -> Dict[str, str]:
    if not group_nodes:
        return {"name": "", "type": "ENTITY", "description": ""}

    names = [n.name for n in group_nodes if n.name]
    types = [n.type for n in group_nodes if n.type]
    descriptions = [n.description for n in group_nodes if n.description]

    canonical_name = Counter(names).most_common(1)[0][0] if names else ""
    canonical_type = Counter(types).most_common(1)[0][0] if types else "ENTITY"

    unique_descriptions = list(dict.fromkeys(descriptions))
    if not unique_descriptions:
        canonical_description = ""
    elif len(unique_descriptions) == 1:
        canonical_description = unique_descriptions[0]
    else:
        canonical_description = _summarize_descriptions(
            unique_descriptions,
            model_name=description_model_name,
            provider=description_provider,
            **description_llm_kwargs,
        )

    return {
        "name": canonical_name,
        "type": canonical_type,
        "description": canonical_description,
    }


def build_representatives(
    nodes_by_group: Dict[str, List[ExtractedNode]],
    existing_nodes: Optional[List[Dict[str, Any]]] = None,
    existing_node_threshold: float = 0.05,
    embedding_model_name: str = "embeddinggemma:latest",
    embedding_provider: str = "ollama",
    description_model_name: str = "gemma3:12b",
    description_provider: str = "ollama",
    embedding_kwargs: Optional[Dict[str, Any]] = None,
    description_llm_kwargs: Optional[Dict[str, Any]] = None,
) -> List[RepresentativeNodeResult]:
    """Build representative node descriptors for each group."""

    embedding_kwargs = embedding_kwargs or {}
    description_llm_kwargs = description_llm_kwargs or {}

    existing_embedded_nodes = _build_existing_nodes_with_embeddings(
        existing_nodes or [],
        embedding_model_name=embedding_model_name,
        embedding_provider=embedding_provider,
        **embedding_kwargs,
    )

    representatives: List[RepresentativeNodeResult] = []
    for group_id, group_nodes in nodes_by_group.items():
        existing_node = match_group_to_existing_nodes(
            group_nodes,
            existing_embedded_nodes,
            threshold=existing_node_threshold,
        )

        if existing_node is not None:
            representatives.append(
                RepresentativeNodeResult(
                    group_id=group_id,
                    name=existing_node.name,
                    type=existing_node.type,
                    description=existing_node.description,
                    mapping_node_ids=[n.node_id for n in group_nodes],
                )
            )
            continue

        metadata = aggregate_group_metadata(
            group_nodes,
            description_model_name=description_model_name,
            description_provider=description_provider,
            **description_llm_kwargs,
        )
        representatives.append(
            RepresentativeNodeResult(
                group_id=group_id,
                name=metadata["name"],
                type=metadata["type"],
                description=metadata["description"],
                mapping_node_ids=[n.node_id for n in group_nodes],
            )
        )

    return representatives


def apply_representatives_to_relationships(
    document_relationships,
    representatives: List[RepresentativeNodeResult],
):
    """Rewrite relationship head/tail fields with representative node values."""
    map_by_original_node_id: Dict[str, RepresentativeNodeResult] = {}
    for representative in representatives:
        for mapped_node_id in representative.mapping_node_ids:
            map_by_original_node_id[mapped_node_id] = representative

    for relationships in document_relationships.values():
        for relationship in relationships:
            head_rep = map_by_original_node_id.get(relationship.head_id)
            if head_rep is not None:
                relationship.head = head_rep.name
                relationship.head_type = head_rep.type
                relationship.head_description = head_rep.description

            tail_rep = map_by_original_node_id.get(relationship.tail_id)
            if tail_rep is not None:
                relationship.tail = tail_rep.name
                relationship.tail_type = tail_rep.type
                relationship.tail_description = tail_rep.description

    return document_relationships
