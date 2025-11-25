import os
from dotenv import load_dotenv
from google.genai import client

from retrieve import retrieve

def get_client():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")
    return client.Client(api_key=api_key)

def relevance_agent(query, top_k=3):
    """
    Uses FAISS + embeddings (retrieve.py) to get the most relevant chunks.
    This is your retrieval / relevance filter.
    """
    chunks = retrieve(query, top_k=top_k)
    return chunks  # list of dicts: {chunk_id, text, distance}


def generator_agent(query, chunks, cl):
    """
    Generates a draft answer using ONLY the retrieved chunks as context.
    """
    context_text = ""
    for item in chunks:
        context_text += f"Chunk (id={item['chunk_id']}):\n{item['text']}\n\n"

    prompt = f"""
You are an assistant answering questions based ONLY on the provided context.

Context:
{context_text}

Question:
{query}

Task:
1. Use only the information in the context to answer the question.
2. If the answer is incomplete, say so.
3. If the answer is not in the context, say "I do not know based on the provided document."

Draft Answer:
"""

    response = cl.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text.strip()

def fact_check_agent(query, chunks, draft_answer, cl):
    """
    Checks the draft answer against the context.
    Corrects any unsupported claims and removes hallucinations.
    """
    context_text = ""
    for item in chunks:
        context_text += f"Chunk (id={item['chunk_id']}):\n{item['text']}\n\n"

    prompt = f"""
You are a strict fact-checking assistant.

Your job:
- Compare the DRAFT ANSWER to the CONTEXT.
- Keep only statements that are supported by the context.
- If any part of the draft answer is not clearly supported, remove or correct it.
- If the question cannot be answered from the context, reply with:
  "I do not know based on the provided document."

CONTEXT:
{context_text}

QUESTION:
{query}

DRAFT ANSWER:
{draft_answer}

Now produce a FINAL ANSWER that:
- Is fully consistent with the context.
- Does NOT invent new facts.
- Clearly says "I do not know based on the provided document." if needed.

FINAL ANSWER:
"""

    response = cl.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text.strip()

def self_correcting_rag(query, top_k=3):
    """
    Full pipeline:
    1. Relevance Agent: retrieve relevant chunks.
    2. Generator Agent: draft an answer.
    3. Fact-Check Agent: verify and correct the draft.
    """
    cl = get_client()

    # 1) Relevance Agent
    chunks = relevance_agent(query)

    # 2) Generator Agent
    draft = generator_agent(query, chunks, cl)

    # 3) Fact-Check Agent
    final = fact_check_agent(query, chunks, draft, cl)

    return draft, final ,chunks

if __name__ == "__main__":
    user_query = "What is the main goal or conclusion of this paper?"

    draft_answer, final_answer = self_correcting_rag(user_query,top_k=3)

    print("\n=== DRAFT ANSWER ===\n")
    print(draft_answer)

    print("\n=== FINAL (SELF-CORRECTED) ANSWER ===\n")
    print(final_answer)
