from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

from app import config


@lru_cache(maxsize=1)
def get_store() -> InMemoryVectorStore:
    if not config.INDEX_PATH.exists():
        raise FileNotFoundError("No index found. Run:  python -m app.ingest")
    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    return InMemoryVectorStore.load(str(config.INDEX_PATH), embeddings)


def retrieve(query: str, k: int | None = None) -> list[Document]:
    return get_store().similarity_search(query, k=k or config.TOP_K)