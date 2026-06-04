import chromadb

CHROMADB_PATH = "./chroma_db"


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