"""Load documents -> chunk -> embed -> store in Chroma.

Run:  python -m app.ingest
"""
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import config

SUPPORTED = {".md", ".txt", ".pdf"}


def sync_from_s3() -> None:
    """If S3_BUCKET is set, download documents into DOCS_DIR (cloud storage step)."""
    if not config.S3_BUCKET:
        return
    import boto3

    s3 = boto3.client("s3", endpoint_url=config.S3_ENDPOINT_URL)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=config.S3_BUCKET, Prefix=config.S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if Path(key).suffix.lower() not in SUPPORTED:
                continue
            s3.download_file(config.S3_BUCKET, key, str(config.DOCS_DIR / Path(key).name))
            count += 1
    print(f"Downloaded {count} file(s) from s3://{config.S3_BUCKET}/{config.S3_PREFIX}")


def load_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(config.DOCS_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED:
            continue
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:
            text = path.read_text(encoding="utf-8")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def build_index() -> int:
    sync_from_s3()
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No documents found in {config.DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # Rebuild from scratch so re-running never creates duplicates
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)

    embeddings = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    store = Chroma(
        collection_name="mydesk",
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    store.add_documents(chunks)
    print(f"Indexed {len(docs)} document(s) -> {len(chunks)} chunks")
    return len(chunks)


if __name__ == "__main__":
    build_index()
