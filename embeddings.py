"""
Step 2: Embeddings.

Turn text into vectors using Voyage AI, so that semantic similarity can later
be measured as vector distance/similarity.
"""

import os

import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

MODEL = "voyage-3"

_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])


def embed_documents(texts: list[str]) -> np.ndarray:
    """
    Embed a list of chunks that will be stored/searched over.

    Returns an (n, d) numpy array: one d-dimensional vector per input text.
    """
    result = _client.embed(texts, model=MODEL, input_type="document")
    return np.array(result.embeddings)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single search query.

    Uses input_type="query" rather than "document" -- Voyage's models are
    trained asymmetrically, so queries and documents are embedded slightly
    differently even though they share the same vector space. Using the
    right one on each side improves retrieval quality.

    Returns a (d,) numpy array.
    """
    result = _client.embed([text], model=MODEL, input_type="query")
    return np.array(result.embeddings[0])


if __name__ == "__main__":
    doc_vecs = embed_documents(["The sky is blue.", "Cats are mammals."])
    query_vec = embed_query("What color is the sky?")

    print("doc embeddings shape:", doc_vecs.shape)
    print("query embedding shape:", query_vec.shape)
