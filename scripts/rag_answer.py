import os
from dotenv import load_dotenv
from google.genai import client

from retrieve import retrieve

def build_prompt(query, retrieved_chunks):
    context_text = ""

    for item in retrieved_chunks:
        context_text += f"Chunk:\n{item['text']}\n\n"

    prompt = f"""
You are a factual assistant. Use ONLY the information provided in the context to answer the question.

If the answer is not found in the context, say "I do not know based on the provided document."

Context:
{context_text}

Question:
{query}

Answer:
"""
    return prompt

def rag_answer(query):
    # Load API key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    cl = client.Client(api_key=api_key)

    # Step 1: Retrieve top chunks
    chunks = retrieve(query)

    # Step 2: Build prompt using retrieved chunks
    prompt = build_prompt(query, chunks)

    # Step 3: Generate response from Gemini
    response = cl.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text

if __name__ == "__main__":
    query = "What is the main conclusion of the paper?"
    answer = rag_answer(query)

    print("\nRAG Answer:\n")
    print(answer)
