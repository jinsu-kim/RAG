import os
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

MODEL_MAPPING = {"bge-m3": "BAAI/bge-m3", "kure-v1": "nlpai-lab/KURE-v1", "openai": "text-embedding-3-small"}


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