import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pdf_chunks"

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query, top_k=3):
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    query_vec = embedder.encode([query])[0]
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True
    )
    return search_result

def main():
    
    query = input("Enter your query: ")
    results = retrieve(query)
    print(f"\nTop {len(results)} results:\n")
    for i, res in enumerate(results, 1):
        payload = res.payload
        print(f"Result {i}:")
        print(f"Chunk: {payload.get('chunk', '')[:500]}{'...' if len(payload.get('chunk', '')) > 500 else ''}")
        figures = payload.get('figures', [])
        if figures:
            print(f"Figures: {figures}")
        print("-"*40)

if __name__ == "__main__":
    main() 