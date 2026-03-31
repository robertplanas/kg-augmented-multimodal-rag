import tqdm
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.models import LLMModel

LOGGER = logging.getLogger(__name__)
_THREAD_LOCAL = threading.local()


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
    model_name="gemma3:12b",
    provider="ollama",
    max_workers=1,
    **llm_kwargs,
):
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

    return texts, images, tables
