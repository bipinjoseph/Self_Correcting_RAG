import os
import json

import numpy as np
import faiss
from dotenv import load_dotenv
from google.genai import client

CHUNKS_PATH = "data/chunks.jsonl"
INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/chunk_metadata.json"
EMBED_MODEL = "text-embedding-004"  # from my  model list
def load_chunks(path: str = CHUNKS_PATH):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks.append(record)
    return chunks

def main():
    # 1) Load API key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("No API key found. Check your .env file.")
        return

    # 2) Init Gemini client
    cl = client.Client(api_key=api_key)

    # 3) Load chunks from file
    records = load_chunks()
    print("Loaded", len(records), "chunks from", CHUNKS_PATH)

    embeddings = []
    metadata = []

    for record in records:
        text = record["text"]

        # Call Gemini Embedding API
        response = cl.models.embed_content(
            model=EMBED_MODEL,
            contents=text
        )

        # Extract the vector from the response
        vector = response.embeddings[0].values

        embeddings.append(vector)
        metadata.append({
            "chunk_id": record["chunk_id"],
            "text": text
        })

    embed_array = np.array(embeddings).astype("float32")
    print("Embedding array shape:", embed_array.shape)


    # We will add embedding + FAISS code here later
    # For now we are just checking that loading works.
    # -----------------------------
    # Build FAISS index
    # -----------------------------
    dimension = embed_array.shape[1]   # size of embedding (e.g., 768)
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to FAISS
    index.add(embed_array)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print("FAISS index saved to", INDEX_PATH)

    # -----------------------------
    # Save metadata (chunk_id + text)
    # -----------------------------
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Metadata saved to", METADATA_PATH)



if __name__ == "__main__":
    main()
