import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from google.genai import client

INDEX_PATH = "data/faiss.index"
METADATA_PATH = "data/chunk_metadata.json"
EMBED_MODEL = "text-embedding-004"

def load_index_and_metadata():
    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata

def embed_query(query, cl):
    # Call Gemini embedding model
    response = cl.models.embed_content(
        model=EMBED_MODEL,
        contents=query
    )

    # Extract vector
    vector = response.embeddings[0].values
    return np.array(vector).astype("float32")

def search_faiss(query_vector, index, top_k=3):
    # FAISS expects shape: (1, dimensions)
    query_vector = np.reshape(query_vector, (1, -1))

    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]

def retrieve(query, top_k=3):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    cl = client.Client(api_key=api_key)

    # Load FAISS + metadata
    index, metadata = load_index_and_metadata()

    # Embed user query
    query_vector = embed_query(query, cl)

    # Search FAISS
    distances, indices = search_faiss(query_vector, index, top_k=top_k)

    # Build results
    results = []
    for idx, dist in zip(indices, distances):
        results.append({
            "chunk_id": metadata[idx]["chunk_id"],
            "text": metadata[idx]["text"],
            "distance": float(dist)
        })

    return results

if __name__ == "__main__":
    query = "What is the conclusion of the paper?"
    results = retrieve(query)

    print("\nTop Results:")
    for r in results:
        print("\nChunk ID:", r["chunk_id"])
        print("Distance:", r["distance"])
        print("Text:", r["text"][:300], "...")
