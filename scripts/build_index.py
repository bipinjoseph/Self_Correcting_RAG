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

    # We will add embedding + FAISS code here later
    # For now we are just checking that loading works.


if __name__ == "__main__":
    main()
