"""
Step 3: Vector store.

A minimal in-memory store: hold chunk texts alongside their embedding
vectors, and support "find the k chunks most similar to this query vector."
"""

import numpy as np


class VectorStore:
    def __init__(self):
        self.texts: list[str] = []
        self.vectors: np.ndarray | None = None  # (n, d)

    def add(self, texts: list[str], vectors: np.ndarray) -> None:
        """Add a batch of chunks and their corresponding embedding vectors."""
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.texts.extend(texts)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """
        Return the k stored chunks most similar to `query_vector`, as
        (text, similarity_score) pairs sorted by descending similarity.
        """
        if self.vectors is None or len(self.texts) == 0:
            return []

        # Cosine similarity between query_vector and every stored vector,
        # vectorized: dot product over the norms.
        doc_norms = np.linalg.norm(self.vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        similarities = (self.vectors @ query_vector) / (doc_norms * query_norm)

        k = min(k, len(self.texts))
        # argsort ascending, take the last k, then reverse for descending order.
        top_k_idx = np.argsort(similarities)[-k:][::-1]

        return [(self.texts[i], float(similarities[i])) for i in top_k_idx]


if __name__ == "__main__":
    from chunking import chunk_text
    from embeddings import embed_documents, embed_query

    with open("data/sample.txt") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=300, overlap=50)
    vectors = embed_documents(chunks)

    store = VectorStore()
    store.add(chunks, vectors)

    query = "How long was the first flight?"
    query_vec = embed_query(query)
    results = store.search(query_vec, k=3)

    print(f"Query: {query}\n")
    for text, score in results:
        print(f"[{score:.3f}] {text[:120]}...")
        print()
