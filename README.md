# NCERT RAG (Retrieval Augmented Generation)

This project implements a semantic search system for NCERT textbook content using Qdrant vector database and sentence transformers. It processes markdown files containing textbook content, including text and image references, and enables semantic search across the content.

## Project Structure

```
ncert-rag/
├── output-pdf-md/           # Directory containing markdown files from PDFs
│   ├── lemh101/            # Chapter-specific directories
│   │   └── output.md/
│   │       ├── output.md   # Markdown content
│   │       └── figures/    # Images referenced in the markdown
│   └── lemh102/
├── qdrant.py               # Script for processing and uploading content to Qdrant
├── retriever.py            # Script for semantic search queries
└── requirements.txt        # Python dependencies
```

## Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Qdrant credentials:
```
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Usage

### Processing and Uploading Content

The `qdrant.py` script handles:
- Processing markdown files from the `output-pdf-md` directory
- Extracting text content and image references
- Creating text chunks with proper context
- Computing embeddings using sentence-transformers
- Uploading the chunks to Qdrant

To process and upload content:
```bash
python qdrant.py
```

### Searching Content

The `retriever.py` script provides semantic search functionality:
- Uses the same embedding model for query encoding
- Performs similarity search in Qdrant
- Returns relevant text chunks with their associated metadata

To search content:
```bash
python retriever.py
```

When prompted, enter your query and the system will return the most relevant chunks along with any associated image references.

## Technical Details

- **Embedding Model**: Uses 'all-MiniLM-L6-v2' from sentence-transformers
- **Vector Database**: Qdrant Cloud
- **Chunk Size**: 1000 characters with 200 character overlap
- **Vector Size**: 384 dimensions
- **Distance Metric**: Cosine similarity

## Data Structure

Each chunk in Qdrant contains:
- `chunk_id`: Unique identifier
- `chunk`: The actual text content
- `figures`: List of paths to referenced images (images are not stored in Qdrant)
- Vector embedding of the text content (384 dimensions)

## Notes

- Images are not stored in Qdrant, only their file paths are stored
- The system maintains the relationship between text chunks and their associated images
- Chunks are created with overlap to maintain context across chunk boundaries
- The semantic search uses the same embedding model for both indexing and querying 