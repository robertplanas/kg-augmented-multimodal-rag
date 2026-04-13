from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


class EmbeddingModel:
    """Initializes an embedding backend for local/Ollama, Gemini, or OpenAI models."""

    def __init__(self, model_name, provider="ollama", **kwargs):
        self.model_name = model_name
        self.provider = provider.lower()
        self.model = self._build_model(**kwargs)

    def _normalize_provider_kwargs(self, kwargs):
        normalized_kwargs = dict(kwargs)
        if self.provider == "openai" and "api_key" in normalized_kwargs:
            normalized_kwargs.setdefault(
                "openai_api_key", normalized_kwargs.pop("api_key")
            )
        elif self.provider in {"gemini", "google"} and "api_key" in normalized_kwargs:
            normalized_kwargs.setdefault(
                "google_api_key", normalized_kwargs.pop("api_key")
            )
        return normalized_kwargs

    def _build_model(self, **kwargs):
        kwargs = self._normalize_provider_kwargs(kwargs)
        if self.provider == "ollama":
            return OllamaEmbeddings(model=self.model_name, **kwargs)

        if self.provider == "openai":
            return OpenAIEmbeddings(model=self.model_name, **kwargs)

        if self.provider in {"gemini", "google"}:
            return GoogleGenerativeAIEmbeddings(model=self.model_name, **kwargs)

        if self.provider in {"local", "huggingface"}:
            return HuggingFaceEmbeddings(model_name=self.model_name, **kwargs)

        raise ValueError(
            "Unsupported embedding provider. "
            "Expected one of: ollama, openai, gemini, google, local, huggingface."
        )

    def as_langchain_embedding(self):
        return self.model

    def get_embedding(self, text):
        return self.model.embed_query(text)

    def embed_query(self, text):
        return self.model.embed_query(text)

    def embed_documents(self, texts):
        return self.model.embed_documents(texts)


class LLMModel:
    """Initializes a language model backend for local/Ollama, Gemini, or OpenAI models."""

    def __init__(self, model_name, provider="ollama", **kwargs):
        self.model_name = model_name
        self.provider = provider.lower()
        self.model = self._build_model(**kwargs)

    def _normalize_provider_kwargs(self, kwargs):
        normalized_kwargs = dict(kwargs)
        if self.provider == "openai" and "api_key" in normalized_kwargs:
            normalized_kwargs.setdefault(
                "openai_api_key", normalized_kwargs.pop("api_key")
            )
        elif self.provider in {"gemini", "google"} and "api_key" in normalized_kwargs:
            normalized_kwargs.setdefault(
                "google_api_key", normalized_kwargs.pop("api_key")
            )
        return normalized_kwargs

    def _build_model(self, **kwargs):
        kwargs = self._normalize_provider_kwargs(kwargs)
        if self.provider == "ollama":
            return ChatOllama(model=self.model_name, **kwargs)

        if self.provider == "openai":
            return ChatOpenAI(model=self.model_name, **kwargs)

        if self.provider in {"gemini", "google"}:
            return ChatGoogleGenerativeAI(model=self.model_name, **kwargs)

        if self.provider in {"local", "huggingface"}:
            task = kwargs.pop("task", "text-generation")
            return HuggingFacePipeline.from_model_id(
                model_id=self.model_name,
                task=task,
                **kwargs,
            )

        raise ValueError(
            "Unsupported LLM provider. "
            "Expected one of: ollama, openai, gemini, google, local, huggingface."
        )

    def as_langchain_llm(self):
        return self.model
