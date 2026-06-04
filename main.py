import argparse
import json
import re

from pathlib import Path
from dotenv import load_dotenv

from pdf_loader import load_all_pdfs
from chunker import chunk_pages
from vector_store import get_collection, get_indexed_sources, save_chunks
from embedding import embed, load_embedder


load_dotenv()


def preprocess_pages(pages: list[dict]) -> list[dict]:

    processed = []
    for page in pages:
        text = page["text"]

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        page["text"] = text

        processed.append(page)

    return processed


def build_index(args, target_files):

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


def ensure_index(args, collection_name:str):

    collection = get_collection(collection_name)
    if collection.count() == 0:
        print("Empty ChromaDB")
        build_index(args)
        return None

    else:
        indexed_sources = get_indexed_sources(collection_name)
        actual_files = {f.name for f in Path("./data/raw").glob("*.pdf")}
        new_files = actual_files - indexed_sources

        if new_files:
            print(f"New Docs: {new_files}")
            build_index(args, target_files=new_files)


def answer_question(args):

    collection_name = f"spri_{args.model.replace('-', '_')}"

    embedder    = load_embedder(args.model)
    query_embed = embed(texts=args.query, model_name=args.model, embedder=embedder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",      type=str, required=True)
    parser.add_argument("--model",      type=str, default="kure-v1", choices=["kure-v1", "bge-m3", "openai"])
    parser.add_argument("--top-k",      type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap",    type=int, default=50)

    args = parser.parse_args()

    ensure_index(args)

    answer_question(args)


if __name__ == "__main__":
    main()