import argparse
import json
import os
import re
import chromadb
from pathlib import Path
from dotenv import load_dotenv

from pdf_loader import load_all_pdfs

load_dotenv()
CHROMADB_PATH = "./chroma_db"


def get_chunk_count():


def get_indexed_sources(collection_name:str) -> set[str]:

    client     = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = client.get_or_create_collection(name=collection_name)
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


def run_index(args, target_files):

    pages           = load_all_pdfs(data_dir="./data/raw", target_files=target_files)
    processed_pages = preprocess_pages(pages)

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

    args = parser.parse_args()

    ensure_index(args)

    # 질문 실행
    run_ask(args)


if __name__ == "__main__":
    main()