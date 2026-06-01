import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()



def get_chunk_count():


def get_indexed_sources():


def run_index():


def run_ask():


def ensure_index(args):

    if get_chunk_count() == 0:
        print("ChromaDB가 비어있습니다.")
        run_index(args)
        return None

    else:
        indexed_sources = get_indexed_sources()
        actual_files = {f.name for f in Path("./data/raw").glob("*.pdf")}
        new_files = actual_files - indexed_sources
        if new_files:
            print(f"새 문서 감지: {new_files}")
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