"""
Step 5: Wire it all together.

Index a document once, then answer questions against it in a loop:

    load -> chunk -> embed -> store
    query -> embed -> search -> generate
"""

from chunking import chunk_text
from embeddings import embed_documents, embed_query
from generation import generate_answer
from vectorstore import VectorStore

DOC_PATH = "data/sample.txt"
TOP_K = 3


def build_index(path: str) -> VectorStore:
    with open(path) as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=300, overlap=50)
    vectors = embed_documents(chunks)

    store = VectorStore()
    store.add(chunks, vectors)
    print(f"Indexed {len(chunks)} chunks from {path}")
    return store


def answer(store: VectorStore, query: str) -> str:
    query_vec = embed_query(query)
    results = store.search(query_vec, k=TOP_K)
    retrieved_chunks = [text for text, _score in results]
    return generate_answer(query, retrieved_chunks)


def main():
    store = build_index(DOC_PATH)

    print("Ask a question (empty line to quit).\n")
    while True:
        query = input("> ").strip()
        if not query:
            break

        response = answer(store, query)
        print(f"\n{response}\n")


if __name__ == "__main__":
    main()
