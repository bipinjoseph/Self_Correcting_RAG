Self-Correcting Retrieval-Augmented Generation (RAG) System

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline enhanced with a self-correcting verification layer. The system retrieves relevant information from source documents, generates an initial draft answer, and then validates that answer against the retrieved context. This multi-agent workflow ensures that the final response is factual, grounded, and stable, reducing hallucinations commonly seen in standard LLM-based systems.

The project uses FAISS for retrieval, Gemini models for embeddings and text generation, and Streamlit for providing an interactive user interface.

1. Overview

The primary objective of this project is to design a robust and explainable RAG system suitable for academic, enterprise, and research environments.
A conventional RAG pipeline retrieves contextual text and uses it to generate an answer. However, this system goes further by introducing a fact-checking agent that ensures the generated answer does not contain unsupported statements.

This is achieved through a three-agent architecture:

Retrieval Agent
Locates the most relevant document chunks using FAISS similarity search.

Generator Agent
Produces a draft answer using only the retrieved content.

Fact-Check Agent
Validates the draft answer, removes unsupported claims, and generates a corrected final answer.

2. Features

PDF text ingestion and chunking

Embedding generation using Gemini (text-embedding-004)

FAISS vector index creation

Adjustable top-k retrieval

Three-agent self-correcting pipeline

Detailed chunk metadata and distance scores

Streamlit UI with interactive controls

Clear separation between draft answer and corrected final answer

Modular and extendable architecture

3. Architecture

The system processes a user query through multiple stages to ensure accuracy and consistency.

User Query
     │
     ▼
Retrieval Agent
(FAISS similarity search using embeddings)
     │
     ▼
Generator Agent
(draft answer based only on retrieved chunks)
     │
     ▼
Fact-Check Agent
(validates and corrects draft answer)
     │
     ▼
Final Answer
(grounded, corrected response)


This design prevents unsupported claims and ensures high factual integrity.

4. Project Structure
Self_RAG_ProJECT/
│
├── apps/
│   └── app.py                     # Streamlit UI
│
├── scripts/
│   ├── build_index.py             # Embeddings + FAISS index creation
│   ├── retrieve.py                # Semantic retrieval logic
│   ├── rag_answer.py              # Standard RAG (no verification)
│   └── self_correcting_rag.py     # Multi-agent self-correcting RAG
│
├── data/
│   ├── pdfs/                      # PDF files used for indexing
│   ├── chunks.jsonl               # Chunked document text
│   ├── faiss.index                # Vector index created from embeddings
│   └── chunk_metadata.json        # Metadata for each chunk
│
├── .gitignore
├── README.md
└── requirements.txt

5. Installation

Install required dependencies:

pip install -r requirements.txt


Add your Gemini API key to a .env file in the project root:

GOOGLE_API_KEY=your_api_key_here

6. Usage
Step 1: Add a PDF

Place your document inside:

data/pdfs/

Step 2: Build embeddings and FAISS index
python scripts/build_index.py


This loads chunks, embeds them, and generates:

faiss.index

chunk_metadata.json

Step 3: Test retrieval
python scripts/retrieve.py


This will display the top-k retrieved chunks and their similarity scores.

Step 4: Run the self-correcting RAG pipeline
python scripts/self_correcting_rag.py


You will see:

Draft answer

Final corrected answer

Retrieved chunks

Step 5: Launch the Streamlit UI
streamlit run apps/app.py


Features in the UI:

Adjustable top-k retrieval via sidebar

Toggle for showing draft answer

Toggle for showing retrieved chunks

Interactive question input

Final answer generated through verification

Access in browser:

http://localhost:8501

7. Technologies Used

Python

FAISS for vector search

Gemini models for embeddings and answer generation

Streamlit for the interactive interface

JSONL for chunk storage

Dotenv for API configuration

8. Future Enhancements

Planned improvements include:

Upload new PDF files directly from UI

Automatic FAISS rebuild when new data is uploaded

Multi-document retrieval and cross-document ranking

Provenance highlighting (highlighting the chunk that supports each part of the answer)

Confidence scoring for both generation and verification

A class-based RAG engine for cleaner modularity

Chat history within the UI

PDF viewer integrated in the interface

9. Author

Bipin Joseph
Master of Computer Applications (AI & ML)
2025–2026