from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app import config


@lru_cache(maxsize=1)
def get_store() -> Chroma:
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    return Chroma(
        collection_name="mydesk",
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def retrieve(query: str, k: int | None = None) -> list[Document]:
    return get_store().similarity_search(query, k=k or config.TOP_K)
