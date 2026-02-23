import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from utils.prompts import RAG_SYSTEM_PROMPT

LOGGER = logging.getLogger(__name__)


def parse_documents(docs):
    b64_images = []
    text_content = []

    for doc in docs:
        # 1. Handle potential bytes from the store
        raw_content = doc.decode("utf-8") if isinstance(doc, bytes) else doc

        try:
            # 2. Try to parse the JSON we saved
            data = json.loads(raw_content)
            content = data.get("content", "")
            data_type = data.get("type", "")

        except (json.JSONDecodeError, TypeError):
            # Fallback if it's old data or not JSON
            content = raw_content
            data_type = "text"  # or your length heuristic

        # 3. Route based on the unpacked metadata
        if data_type == "image":
            b64_images.append(content)
        else:
            text_content.append(content)

    return {"images": b64_images, "texts": text_content}


def build_prompt(input_dict, system_prompt=RAG_SYSTEM_PROMPT):
    # Organize text context with clear markers
    texts = input_dict["context"].get("texts", [])
    formatted_context = "\n\n".join([f"[Source {i}]: {t}" for i, t in enumerate(texts)])

    # Construct the multimodal content list
    user_content = [
        {
            "type": "text",
            "text": f"Use the following context to answer the question.\n\nContext:\n{formatted_context}\n\nQuestion: {input_dict['question']}",
        }
    ]

    # Append images with base64 strings
    for img_b64 in input_dict["context"].get("images", []):
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            }
        )

    return [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]


def rag_chain(retriever, model="gemma3:12b"):
    chain = (
        {
            "context": retriever | RunnableLambda(parse_documents),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOllama(model=model)
        | StrOutputParser()
    )
    return chain
