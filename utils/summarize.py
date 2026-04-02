import tqdm
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from utils.models import LLMModel

LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()

PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def _resolve_summary_config(
    model_name: Optional[str],
    provider: Optional[str],
    max_workers: Optional[int],
    llm_kwargs,
):
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=project_root / ".env")
    load_dotenv(dotenv_path=project_root / ".venv" / ".env")

    resolved_provider = (
        provider or os.getenv("SUMMARY_LLM_PROVIDER") or "ollama"
    ).lower()
    resolved_model_name = model_name or os.getenv("SUMMARY_LLM_MODEL") or "gemma3:12b"

    if max_workers is None:
        env_workers = os.getenv("SUMMARY_LLM_MAX_WORKERS", "1")
        try:
            resolved_max_workers = int(env_workers)
        except ValueError:
            raise ValueError(
                f"Invalid SUMMARY_LLM_MAX_WORKERS value: {env_workers}. Expected integer."
            ) from None
    else:
        resolved_max_workers = max_workers

    api_key_env = PROVIDER_API_KEY_ENV.get(resolved_provider)
    if api_key_env and "api_key" not in llm_kwargs:
        api_key = os.getenv(api_key_env)
        if api_key:
            llm_kwargs["api_key"] = api_key
        else:
            raise ValueError(
                f"Missing API key for summary provider '{resolved_provider}'. "
                f"Set {api_key_env} in your environment or in {project_root / '.env'}."
            )

    return resolved_model_name, resolved_provider, resolved_max_workers, llm_kwargs


def _get_thread_llm(model_name, provider, llm_kwargs):
    key = (model_name, provider, tuple(sorted(llm_kwargs.items())))
    if getattr(_THREAD_LOCAL, "llm_key", None) != key:
        _THREAD_LOCAL.llm = LLMModel(
            model_name=model_name, provider=provider, **llm_kwargs
        )
        _THREAD_LOCAL.llm_key = key
    return _THREAD_LOCAL.llm


def _summarize_one(obj, summarize_method, model_name, provider, llm_kwargs):
    llm = _get_thread_llm(model_name, provider, llm_kwargs)
    getattr(obj, summarize_method)(llm=llm)


def _run_summary_batch(
    objects,
    summarize_method,
    batch_label,
    model_name,
    provider,
    llm_kwargs,
    max_workers,
):
    if len(objects) == 0:
        LOGGER.info("No %s to summarize", batch_label)
        return

    LOGGER.info("Summarizing %s", batch_label)

    if max_workers is None or max_workers <= 1:
        for obj in tqdm.tqdm(objects):
            _summarize_one(obj, summarize_method, model_name, provider, llm_kwargs)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _summarize_one,
                obj,
                summarize_method,
                model_name,
                provider,
                llm_kwargs,
            )
            for obj in objects
        ]

        progress = tqdm.tqdm(total=len(futures))
        try:
            for future in as_completed(futures):
                future.result()
                progress.update(1)
        finally:
            progress.close()


def summarize_objects(
    texts,
    images,
    tables,
    codes,
    model_name=None,
    provider=None,
    max_workers=None,
    **llm_kwargs,
):
    model_name, provider, max_workers, llm_kwargs = _resolve_summary_config(
        model_name=model_name,
        provider=provider,
        max_workers=max_workers,
        llm_kwargs=llm_kwargs,
    )

    _run_summary_batch(
        objects=texts,
        summarize_method="summarize_text",
        batch_label="text chunks",
        model_name=model_name,
        provider=provider,
        llm_kwargs=llm_kwargs,
        max_workers=max_workers,
    )

    _run_summary_batch(
        objects=images,
        summarize_method="summarize_image",
        batch_label="images",
        model_name=model_name,
        provider=provider,
        llm_kwargs=llm_kwargs,
        max_workers=max_workers,
    )

    _run_summary_batch(
        objects=tables,
        summarize_method="summarize_table",
        batch_label="tables",
        model_name=model_name,
        provider=provider,
        llm_kwargs=llm_kwargs,
        max_workers=max_workers,
    )

    _run_summary_batch(
        objects=codes,
        summarize_method="summarize_code",
        batch_label="code blocks",
        model_name=model_name,
        provider=provider,
        llm_kwargs=llm_kwargs,
        max_workers=max_workers,
    )

    return texts, images, tables, codes
