import os
import re
import uuid
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pdf_chunks"

# Load the embedding model once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_figures(chunk):
    # Find all image markdowns: ![](figures/filename.png)
    return re.findall(r'!\[\]\((figures/[^)]+)\)', chunk)

def process_markdown(md_path, chunk_sizes=[1000, 200]):
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_sizes[0], chunk_overlap=chunk_sizes[1])
    chunks = splitter.split_text(text)
    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        figures = extract_figures(chunk)
        chunk_dicts.append({
            "chunk_id": str(uuid.uuid4()),
            "chunk": chunk,
            "figures": figures
        })
    print("Sample chunk dicts (first 3):")
    for c in chunk_dicts[:3]:
        print(json.dumps(c, indent=2, ensure_ascii=False))
    return chunk_dicts

def upload_to_qdrant(chunks, client, collection_name):
    print(f"Uploading {len(chunks)} chunks to Qdrant...")
    texts = [c["chunk"] for c in chunks]
    vectors = embedder.encode(texts, show_progress_bar=True).tolist()
    payloads = [{"chunk_id": c["chunk_id"], "chunk": c["chunk"], "figures": c["figures"]} for c in chunks]
    try:
        client.upload_points(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=c["chunk_id"],
                    vector=v,
                    payload=p
                )
                for c, v, p in zip(chunks, vectors, payloads)
            ]
        )
        print(f"Upload complete for {len(chunks)} chunks.")
    except Exception as e:
        print(f"Error during upsert: {e}")

def print_sample_from_qdrant(client, collection_name, n=3):
    print(f"\nFetching {n} sample points from Qdrant collection '{collection_name}'...")
    points, _ = client.scroll(collection_name=collection_name, limit=n, with_payload=True)
    for i, pt in enumerate(points, 1):
        print(f"Sample {i}:")
        print(json.dumps(pt.payload, indent=2, ensure_ascii=False))
        print("-"*40)

def main():
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Qdrant API Key (prefix): {QDRANT_API_KEY[:8]}...\n")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    # Create collection if not exists (with correct vector size)
    print(f"Checking for collection '{COLLECTION_NAME}'...")
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found. Creating...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    base_dir = "output-pdf-md"
    print("Looking for markdown files to process...")
    for pdf_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, pdf_folder, "output.md")
        md_path = os.path.join(folder_path, "output.md")
        print(f"Checking: {md_path}")
        if os.path.isfile(md_path):
            print(f"Processing {md_path}")
            chunks = process_markdown(md_path)
            print(f"Chunked into {len(chunks)} pieces.")
            upload_to_qdrant(chunks, client, COLLECTION_NAME)
            with open(os.path.join(folder_path, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
    print_sample_from_qdrant(client, COLLECTION_NAME, n=3)

if __name__ == "__main__":
    main()