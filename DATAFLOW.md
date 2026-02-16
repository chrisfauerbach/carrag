# Data Flow: Document Ingestion

What happens, step by step, when you POST a document to Carrag.

---

## The Big Picture

```
You upload a file or URL
        │
        ▼
   ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌───────────────┐
   │  Parse   │────▶│  Chunk   │────▶│  Embed  │────▶│  Store in ES  │
   │  (text)  │     │  (split) │     │ (Ollama)│     │  (bulk index) │
   └─────────┘     └──────────┘     └─────────┘     └───────────────┘
                                                             │
                                                             ▼
                                                     Response with
                                                     document_id
```

There are two entry points — file upload and URL — but they converge into the same pipeline after parsing.

---

## Step 1: The HTTP Request Arrives

### File Upload: `POST /ingest/file`

You send a multipart form with a file attached:

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@my-document.pdf"
```

The route handler (`api/routes/ingest.py:ingest_file`) does three validation checks before anything else:

1. **Extension check** — Is it `.pdf`, `.txt`, `.md`, `.text`, or `.markdown`? If not, you get a 400 error immediately.
2. **Empty file check** — The entire file is read into memory as bytes. If zero bytes, 400 error.
3. **Content check** — After parsing (next step), if there's no extractable text, 400 error.

### URL: `POST /ingest/url`

```bash
curl -X POST http://localhost:8000/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/article"}'
```

The route handler (`api/routes/ingest.py:ingest_url`) fetches the page and extracts text. Same empty-content check applies.

---

## Step 2: Parsing — Extract Raw Text

The goal here is simple: turn whatever you uploaded into a plain text string plus some metadata about where it came from.

### PDF files (`services/parsers/pdf.py`)

Uses PyMuPDF (fitz) to open the bytes as a PDF:

```
Raw PDF bytes
    │
    ▼
Open with fitz ──▶ Iterate pages ──▶ Extract text from each page
                                            │
                                            ▼
                                   Join all pages with "\n\n"
```

- Each page's text is extracted via `page.get_text()`
- Pages with no text (scanned images, blank pages) are skipped
- All non-empty pages are joined with double newlines between them
- Metadata captured: `filename`, `source_type: "pdf"`, `total_pages`, `pages_with_text`

**Output:**
```python
{
    "content": "Page 1 text here...\n\nPage 2 text here...",
    "metadata": {
        "filename": "my-document.pdf",
        "source_type": "pdf",
        "total_pages": 12,
        "pages_with_text": 10
    }
}
```

### Text/Markdown files (`services/parsers/text.py`)

Straightforward — decode the bytes as UTF-8:

```
Raw bytes ──▶ Decode UTF-8 ──▶ Done
```

- Uses `errors="replace"` so bad bytes become `�` instead of crashing
- Metadata captured: `filename`, `source_type: "text"`, `line_count`

### Web pages (`services/parsers/web.py`)

More involved — needs to fetch the page and strip out the HTML noise:

```
URL
 │
 ▼
httpx GET (follows redirects, 30s timeout)
 │
 ▼
Raw HTML
 │
 ├──▶ trafilatura.extract() ──── worked? ──▶ Clean article text
 │                                  │
 │                              failed (None)
 │                                  │
 └──▶ BeautifulSoup fallback ◀─────┘
      Remove: <script>, <style>, <nav>, <footer>, <header>
      Then: soup.get_text(separator="\n", strip=True)
```

- **trafilatura** is tried first — it's purpose-built for extracting article content from web pages and does a good job of ignoring navbars, sidebars, footers, etc.
- If trafilatura returns `None` (can't figure out the main content), **BeautifulSoup** is the fallback — it strips obvious non-content tags and grabs all remaining text.
- The page title is extracted either way (from `<title>` tag).
- Metadata captured: `filename` (set to the URL), `source_type: "web"`, `url`, `title`, `extracted_at` (UTC timestamp)

---

## Step 3: Generate a Document ID

Before entering the shared pipeline, a UUID v4 is generated:

```python
document_id = str(uuid.uuid4())
# e.g., "a3f1b2c4-5678-9def-abcd-ef1234567890"
```

This ID ties all the chunks from this document together. It's how you later list, inspect, or delete the document.

---

## Step 4: Chunking — Split Text Into Pieces

**File:** `services/chunker.py`
**Config:** `CHUNK_SIZE=2000` chars, `CHUNK_OVERLAP=200` chars

The parsed text is usually too long to embed as a single vector (embeddings work best on focused passages). The chunker splits it into overlapping pieces.

### How recursive splitting works

The chunker tries to split on natural boundaries, falling back to smaller boundaries only when needed:

```
Full document text
        │
        ▼
Try splitting on "\n\n" (paragraph breaks)
        │
        ├── Each piece ≤ 2000 chars? ──▶ Keep it
        │
        └── Piece too big? ──▶ Try splitting on "\n" (line breaks)
                                    │
                                    ├── ≤ 2000 chars? ──▶ Keep it
                                    │
                                    └── Still too big? ──▶ Try ". " (sentences)
                                                              │
                                                              └── Still too big? ──▶ Try " " (words)
                                                                                        │
                                                                                        └── STILL too big?
                                                                                            Hard split at 2000 chars
```

**The merging logic:** After splitting on a separator, adjacent small pieces are merged back together until they'd exceed 2000 chars. This avoids creating tiny chunks from short paragraphs.

For example, with three paragraphs of 400, 300, and 500 chars:
- 400 + 300 = 700 (under 2000, merge them)
- 700 + 500 = 1200 (still under 2000, merge again)
- Result: one chunk of 1200 chars instead of three tiny ones

### Overlap tracking

Each chunk records its character position in the original text. The offset advances by `chunk_length - overlap` between chunks:

```
Original text:  [==============================================]

Chunk 0:        [===========]          (chars 0–2000)
Chunk 1:             [===========]     (chars 1800–3800)  ← 200 char overlap
Chunk 2:                  [===========]
```

### Output

```python
[
    {"text": "chunk text...", "document_id": "a3f1b2c4-...", "chunk_index": 0, "char_start": 0,    "char_end": 1847},
    {"text": "chunk text...", "document_id": "a3f1b2c4-...", "chunk_index": 1, "char_start": 1647,  "char_end": 3521},
    {"text": "chunk text...", "document_id": "a3f1b2c4-...", "chunk_index": 2, "char_start": 3321,  "char_end": 4900},
]
```

---

## Step 5: Embedding — Turn Text Into Vectors

**File:** `services/embeddings.py`
**Model:** `nomic-embed-text` (768 dimensions)

Each chunk's text needs to become a vector (list of 768 floats) so Elasticsearch can do similarity search later.

### Batching

Chunks are sent to Ollama in batches of 32 to avoid overwhelming the API:

```
Chunks:  [c0, c1, c2, ... c99]  (100 chunks total)

Batch 1: [c0  – c31]  ──▶ POST /api/embed ──▶ [vec0  – vec31]
Batch 2: [c32 – c63]  ──▶ POST /api/embed ──▶ [vec32 – vec63]
Batch 3: [c64 – c95]  ──▶ POST /api/embed ──▶ [vec64 – vec95]
Batch 4: [c96 – c99]  ──▶ POST /api/embed ──▶ [vec96 – vec99]
```

### The Ollama API call

Each batch sends a POST to Ollama:

```
POST http://ollama:11434/api/embed
{
    "model": "nomic-embed-text",
    "input": ["chunk 0 text...", "chunk 1 text...", ...]
}

Response:
{
    "embeddings": [
        [0.0123, -0.0456, 0.0789, ...],   ← 768 floats for chunk 0
        [0.0111, -0.0222, 0.0333, ...],   ← 768 floats for chunk 1
        ...
    ]
}
```

After all batches complete, there's a 1:1 mapping between chunks and embedding vectors.

---

## Step 6: Store in Elasticsearch

**File:** `services/elasticsearch.py`
**Index:** `carrag_chunks`

### What gets stored per chunk

Each chunk becomes one Elasticsearch document:

```json
{
    "content":     "The actual chunk text...",
    "embedding":   [0.0123, -0.0456, ...],        // 768-dim vector
    "document_id": "a3f1b2c4-5678-9def-...",       // groups chunks together
    "chunk_index": 0,                               // position in document
    "char_start":  0,                               // start offset in original
    "char_end":    1847,                             // end offset in original
    "metadata": {                                    // from the parser
        "filename": "my-document.pdf",
        "source_type": "pdf",
        "total_pages": 12,
        "pages_with_text": 10
    },
    "created_at":  "2026-02-16T15:00:00+00:00"
}
```

### Bulk indexing

All chunks are inserted in a single bulk operation using Elasticsearch's `_bulk` API (via the `async_bulk` helper). This is much faster than inserting one by one.

```
async_bulk(es_client, [action0, action1, action2, ...])
    │
    ▼
Single HTTP request to ES with all documents
    │
    ▼
es_client.indices.refresh()   ← makes documents immediately searchable
```

### The ES index schema

Created once at app startup. The key field is `embedding`:

```
embedding: dense_vector
  ├── dims: 768          (matches nomic-embed-text output)
  ├── index: true        (builds HNSW index for fast kNN search)
  └── similarity: cosine (how vectors are compared)
```

---

## Step 7: Return the Response

After bulk indexing completes, the API returns:

```json
{
    "document_id": "a3f1b2c4-5678-9def-abcd-ef1234567890",
    "filename": "my-document.pdf",
    "chunk_count": 7,
    "status": "ingested"
}
```

You keep the `document_id` to query against this document or delete it later.

---

## Complete Timeline for a PDF Upload

```
Time ──▶

│ POST /ingest/file
│
├─ Validate extension (.pdf ✓)
├─ Read file bytes into memory
├─ PyMuPDF extracts text from all pages
├─ Validate extracted text is not empty
├─ Generate UUID for document_id
│
├─ Chunker splits text
│   └─ Try "\n\n" → "\n" → ". " → " " → hard split
│   └─ Merge small pieces, respect 2000 char limit
│   └─ Track character offsets with 200 char overlap
│
├─ Embed chunks (batches of 32)
│   ├─ Batch 1 → Ollama /api/embed → 768-dim vectors
│   ├─ Batch 2 → Ollama /api/embed → 768-dim vectors
│   └─ ...
│
├─ Bulk index into Elasticsearch
│   └─ One ES doc per chunk: text + vector + metadata
│   └─ Refresh index (makes searchable immediately)
│
└─ Return {document_id, filename, chunk_count, status}
```

---

## What Happens Later: Query Flow

Once documents are stored, querying uses the same embedding model to find relevant chunks:

```
POST /query {"question": "What is X?"}
        │
        ▼
   Embed the question  ──▶  Ollama /api/embed  ──▶  768-dim query vector
        │
        ▼
   kNN search in ES    ──▶  Find top-k chunks with highest cosine similarity
        │                    (default k=5, searches 50 candidates)
        ▼
   Build prompt:
        "Context:
         [Source 1: file.pdf]
         chunk text...
         ---
         [Source 2: file.pdf]
         chunk text...

         Question: What is X?
         Answer based on the context above:"
        │
        ▼
   Ollama /api/generate (llama3.2, stream=false)
        │
        ▼
   Return {answer, sources with scores, model, duration_ms}
```

The LLM is instructed via system prompt to ONLY use the provided context and to cite sources.
