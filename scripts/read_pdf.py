import json
from pypdf import PdfReader

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

file_path = "data/pdfs/bipin synopsis new format pdf.pdf"

reader = PdfReader(file_path)

text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

chunks = chunk_text(text)

print("Total chunks:", len(chunks))
print("First chunk:", chunks[0][:300])

# Save chunks to JSONL file
output_path = "data/chunks.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        record = {
            "chunk_id": f"chunk_{i}",
            "text": chunk
        }
        f.write(json.dumps(record) + "\n")

print("Chunks saved to data/chunks.jsonl")

