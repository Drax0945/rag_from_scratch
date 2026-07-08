"""
Step 1: Chunking.

Split a document into overlapping text chunks that are small enough to embed
and retrieve meaningfully, but large enough to preserve context.
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split `text` into chunks of roughly `chunk_size` characters, where each
    chunk overlaps the previous one by `overlap` characters.

    TODO:
    - Walk through `text` in windows of `chunk_size`, advancing by
      (chunk_size - overlap) each step, so consecutive chunks share
      `overlap` characters at the boundary.
    - Handle the last chunk (it will likely be shorter than chunk_size).
    - Return a list of chunk strings, in order.

    Edge cases to think about:
    - What if `text` is shorter than `chunk_size`?
    - What if `overlap` >= `chunk_size`? (should probably be invalid)
    - Should you split on raw characters, or try to break on whitespace so
      you don't cut a word in half? (character-based is fine for v1 --
      you can improve it later)
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step

    return chunks


if __name__ == "__main__":
    with open("data/sample.txt") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=300, overlap=50)
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c)
        print()
