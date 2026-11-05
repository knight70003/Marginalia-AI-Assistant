# Enterprise Cost-Efficient RAG System & Evaluation Pipeline

A production-ready, low-cost Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **ChromaDB**, and **Google Gemini**. This project serves as a high-performance alternative to expensive managed vector database pods while providing complete evaluation, cost-benefit analysis, and structured benchmark metrics.

---

## Technical Architecture & Operational Workflows

### System Context Diagram

```
+-----------------------------------------------------------------------------------+
|                                 INGESTION PHASE                                   |
|                                                                                   |
|  +---------------+      +------------------------+      +----------------------+  |
|  | ML(doc).pdf   | ---> | PyPDFLoader            | ---> | RecursiveCharacter   |  |
|  | (Source Data) |      | (Document Extraction)  |      | TextSplitter         |  |
|  +---------------+      +------------------------+      | (1000 size / 200 ov) |  |
|                                                         +----------------------+  |
|                                                                    |              |
|  +---------------------+      +-----------------------+            |              |
|  | Persistent Storage  | <--- | Chroma Vector DB      | <----------+              |
|  | (./Chroma_db/)      |      | (Local Vector Store)  |                           |
|  +---------------------+      +-----------------------+                           |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                                 QUERY & RETRIEVAL PHASE                           |
|                                                                                   |
|  +---------------+      +------------------------+      +----------------------+  |
|  | User / UI     | ---> | FastAPI Endpoint       | ---> | HuggingFace          |  |
|  | (Ulapp.py)    |      | (app.py / /api/query)  |      | Embeddings           |  |
|  +---------------+      +------------------------+      | (all-MiniLM-L6-v2)   |  |
|          ^                            ^                 +----------------------+  |
|          |                            |                            |              |
|          | (Formatted Response)       | (Grounded Answer)          v              |
|          |                            |                 +----------------------+  |
|          |                      +-----------+           | Similarity Search    |  |
|          +--------------------- | GROQ      | <-------- | Top-K Chunks         |  |
|                                 | LLM Engine|  (Context)| (k=3 / Metadata)     |  |
|                                 +-----------+           +----------------------+  |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                                 EVALUATION PHASE                                  |
|                                                                                   |
|  +---------------------+      +-----------------------+     +------------------+  |
|  | ground_truth.json   | ---> | evaluation/           | --> | Terminal Benchmark| |
|  | (Expected Pages)    |      | evaluate.py           |     | Output Report    |  |
|  +---------------------+      +-----------------------+     +------------------+  |
|                                           ^                                       |
|  +---------------------+                  |                                       |
|  | results.json        | -----------------+                                       |
|  | (Retrieved Pages)   |                                                          |
|  +---------------------+                                                          |
+-----------------------------------------------------------------------------------+

```

---

## Key Technical Features

* **Cost-Optimized Embedded Vector Store:** Utilizes embedded ChromaDB (`Chroma_db/`) to persist vector embeddings locally, eliminating monthly vector host fees.


* **Idempotent Document Ingestion:** Prevents duplicate vector index creation across backend restarts by checking persistent database state before loading.


* **Robust Character Chunking:** Configured with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`) for seamless context preservation across chunk boundaries.


* **Grounded Answer Engine:** Enforces strict prompt grounding using Google Gemini. Unmatched or out-of-scope queries gracefully return *"No relevant context found"* without hallucinating.


* **Automated IR Evaluation Harness:** Built-in evaluation module (`evaluation/evaluate.py`) calculating Information Retrieval (IR) metrics against annotated ground-truth data.

---

## Project Directory Structure

```text
PROBLEM 1/
├── Assignenv/                  # Python Virtual Environment
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/                # Python binaries & activation scripts
│   └── pyvenv.cfg
├── Chroma_db/                  # Local persistent ChromaDB storage (SQLite & index binaries)
├── evaluation/                 # Evaluation harness sub-package
│   ├── evaluate.py             # IR metrics computation script
│   └── results.json            # Executed test outputs & retrieved metadata
├── .env                        # Environment secrets (GEMINI_API_KEY)
├── .gitignore                  # Git exclusions rules
├── app.py                      # Core FastAPI backend routes & RAG engine
├── ground_truth.json           # Ground-truth mapping dataset for benchmark validation
├── ML(doc).pdf                 # Knowledge base source PDF document
├── questions.txt               # Evaluation test suit prompt list
├── README.md                   # Enterprise system documentation
├── requirements.txt            # Explicit python package dependencies
└── Ulapp.py                    # Web UI application server (FastAPI + HTML/CSS Frontend)

```

---

## Tech Stack & Framework Choices

| Component | Technology | Selection Justification |
| --- | --- | --- |
| **Vector Database** | **ChromaDB (Embedded)**<br> | Zero-cost persistent storage, native metadata filtering support, and zero network-call latency.

 |
| **Embeddings** | `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)

 | Open-source 384-dimensional dense vectors with superior semantic similarity accuracy for PDF docs.

 |
| **LLM Engine** | `ChatGoogleGenerativeAI` (`gemini-2.5-flash`)

 | Sub-second generation response time, cost-efficient token pricing, and high contextual adherence.

 |
| **Backend Framework** | FastAPI & Uvicorn

 | Asynchronous HTTP handling with automatic OpenAPI documentation (`/docs`) and low latency overhead.

 |

---

## Quantitative Evaluation Results

The retrieval module was benchmarked against the gold-standard ground-truth mapping (`ground_truth.json`) using `evaluation/evaluate.py`.

### Benchmark Metrics Summary

```text
======================================================================
OVERALL METRICS
======================================================================
Average Precision@K      : 0.33
Average Recall@K         : 1.00
Average Hit Rate         : 1.00
Average MRR              : 0.92
Average nDCG@K           : 0.94
Average Context Precision: 0.92
======================================================================

```

### Metrics Interpretation

* **Recall@K & Hit Rate (1.00):** The target page containing the exact answer was successfully retrieved in **100% of benchmark test cases**.
* **Mean Reciprocal Rank (MRR = 0.92):** The correct source page was ranked as the top #1 context chunk in almost every test query.
* **nDCG@K (0.94) & Context Precision (0.92):** Confirms ideal relevance ranking, placing primary context at the top of the context window.
* **Precision@K (0.33):** Reflects retrieving $K=3$ total chunks for single-page target documents ($1 \text{ relevant page} / 3 \text{ chunks} = 0.33$).

---

## Cost Analysis (Embedded vs. Managed Vector DB)

Assumptions: $1,000,000$ stored vectors, $100,000$ monthly user search queries.

| Vector Database Architecture | Storage Infra Cost | Query Compute Fee | Estimated Total Monthly Cost |
| --- | --- | --- | --- |
| **Managed DB (e.g., Pinecone Standard Pods)** | ~$70.00 / month | ~$15.00 / month | **~$85.00 / month** |
| **Managed DB (e.g., Qdrant Cloud Cluster)** | ~$50.00 / month | ~$10.00 / month | **~$60.00 / month** |
| **Embedded ChromaDB (This Implementation)**<br> | **$0.00** (Local / EBS Storage) | **$0.00** (In-Process CPU) | **~$0.00 / month** |

---

## Installation & Execution Guide

### 1. Environment Setup

Activate the virtual environment and install required dependencies:

```powershell
# Activate Assignenv (PowerShell)
.\Assignenv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt

```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here

```

### 3. Run Application Server

To launch the API and Web UI application server:

```powershell
uvicorn UIapp:app --reload --host 0.0.0.0 --port 8000

```

* Access Interactive Web UI: `http://localhost:8000`

* Access API Documentation: `http://localhost:8000/docs`


### 4. Run Evaluation Harness

To execute the automated IR evaluation suite:

```powershell
python evaluation/evaluate.py

```

---

## Engineering Discussion & Architecture Trade-Offs

### 1. When to Migrate to a Managed Vector Store?

While embedded ChromaDB is optimal for single-node deployments and low-to-medium vector scale, switching to a managed service (e.g., Pinecone/Qdrant) is recommended under these conditions:

* **Horizontal Scaling (>1M+ Vectors):** When vector index size exceeds host system RAM capacity.
* **Distributed Concurrent Writes:** When multiple stateless backend instances require real-time parallel writes without storage locks.
* **Multi-Tenancy:** When enterprise security requires multi-tenant isolation and strict database-level RBAC.

### 2. Performance Bottlenecks & Recommendations

* **Retrieval Layer:** Achieved perfect recall (`1.00`), but standard similarity search can retrieve redundant overlapping text chunks. Implementing **Maximal Marginal Relevance (MMR)** or Hybrid Search further optimizes context diversity.


* **Generation Layer:** Gemini-based prompt grounding proved highly accurate. The LLM strictly respects context boundaries, eliminating false hallucinations.





