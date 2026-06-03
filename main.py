import argparse
import json
import os
import re
import chromadb
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from pdf_loader import load_all_pdfs

load_dotenv()
CHROMADB_PATH = "./chroma_db"
MODEL_MAPPING = {"bge-m3": "BAAI/bge-m3", "kure-v1": "nlpai-lab/KURE-v1", "openai": "text-embedding-3-small"}


def get_chunk_count():


def get_collection(collection_name:str):
    client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_or_create_collection(name=collection_name)

    return collection

def get_indexed_sources(collection_name:str) -> set[str]:

    collection = get_collection(collection_name)
    result     = collection.get(include=["metadatas"])

    sources = set()
    for metadata in result["metadatas"]:
        if metadata and "source" in metadata:
            sources.add(metadata["source"])

    return sources

def run_ask():


def preprocess_pages(pages: list[dict]) -> list[dict]:

    processed = []
    for page in pages:
        text = page["text"]

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        page["text"] = text

        processed.append(page)

    return processed


def chunk_text(text:str, chunk_size:int, overlap:int) -> list[str]:

    if chunk_size <= overlap or chunk_size < 0 or overlap < 0:
        raise ValueError("Invalid parameters")

    text = text.strip()

    if not text:
        return []

    chunk_list = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunk_list.append(chunk)

        start = end - overlap

    return chunk_list

def chunk_pages(args, pages: list[dict]) ->  list[dict]:

    all_chunks = []

    for page in pages:
        source   = page["source"]
        page_num = page["page"]
        text     = page.get("text", "")

        page_chunks = chunk_text(text=text, chunk_size=args.chunk_size, overlap=args.overlap)

        for idx, texts in enumerate(page_chunks):

            chunk_id = f"{source}_p{page_num}_c{idx}"
            all_chunks.append(
                {"chunk_id":chunk_id, "source":source, "page":page_num, "chunk_text":texts}
            )

    return all_chunks

def save_chunks(chunks: list[dict], embeddings: list[list[float]], collection_name: str):

    collection = get_collection(collection_name)

    ids   = []
    docs  = []
    metas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        docs.append(chunk["chunk_text"])
        metas.append(
                    {"source": chunk["source"], "page": chunk["page"]}
                    )

        collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)


def load_embedder(model_name:str):

    if model_name not in MODEL_MAPPING:
        raise ValueError("Invalid model name")

    elif model_name == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    else:
        SentenceTransformer(MODEL_MAPPING[model_name])


def embed(texts: list[str], model_name: str, embedder, batch_size: int = 32) -> list[list[float]]:

    if not texts:
        return []

    if model_name == "openai":
        res        = embedder.embeddings.create(model=MODEL_MAPPING["openai"], input=texts)
        embeddings = [item.embedding for item in res.data]
        return embeddings

    else:   # bge-m3, kure-v1
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                                    batch_size=batch_size)

        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings


def run_index(args, target_files):

    # preprocessing pages
    pages           = load_all_pdfs(data_dir="./data/raw", target_files=target_files)
    processed_pages = preprocess_pages(pages)

    # chunking pages
    chunked_pages   = chunk_pages(processed_pages)

    # embedding
    embedder        = load_embedder(args.model)

    text_list       = [chunk["chunk_text"] for chunk in chunked_pages]
    embeddings      = embed(texts=text_list, model_name=args.model, embedder=embedder)

    collection_name = f"spri_{args.model.replace('-', '_')}"
    save_chunks(chunks=chunked_pages, embeddings=embeddings, collection_name=collection_name)


def ensure_index(args):

    if get_chunk_count() == 0:
        print("Empty ChromaDB")
        run_index(args)
        return None

    else:
        indexed_sources = get_indexed_sources()
        actual_files = {f.name for f in Path("./data/raw").glob("*.pdf")}
        new_files = actual_files - indexed_sources

        if new_files:
            print(f"New Docs: {new_files}")
            run_index(args, target_files=new_files)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",      type=str, required=True)
    parser.add_argument("--model",      type=str, default="kure-v1", choices=["kure-v1", "bge-m3", "openai"])
    parser.add_argument("--top-k",      type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap",    type=int, default=50)

    args = parser.parse_args()

    ensure_index(args)

    run_ask(args)


if __name__ == "__main__":
    main()