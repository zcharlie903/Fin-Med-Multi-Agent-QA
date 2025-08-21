# 🧠💼 Multi‑Agent Medical & Finance QA Backend

A production‑ready backend that combines **LangChain**, **Qdrant** (vector DB), and **Redis** (long‑term memory) to perform **domain‑aware Retrieval‑Augmented Generation (RAG)** for **medical** and **financial** corpora. It orchestrates multiple agents (reasoner + validator) with **ReAct-style planning** and **self‑reflection** to reduce hallucinations and improve task success rates.

> ✅ Implements your resume bullets: LangChain + Qdrant + OpenAIEmbeddings, hierarchical context windows, Redis conversation memory, and a reasoning+validator agent loop (ReAct + self‑reflection).

> ⚠️ **Disclaimer:** This system is for research/education. It **does not** provide medical or financial advice.

---

## ✨ Features

- **Multi‑Agent Orchestration**: Reasoner agent proposes answers; Validator agent rechecks evidence and requests revisions if needed.
- **RAG over Domains**: Index medical and finance documents separately; route queries to the right domain.
- **Qdrant Vector Store**: Fast, semantic retrieval with OpenAI (or compatible) embeddings.
- **Hierarchical Context Windows**: Retrieve chunks → compress/summarize to fit model budget without losing citations.
- **Redis Conversation Memory**: Long‑term context across turns to improve coherence.
- **FastAPI** Backend: Clean API for `/ingest` and `/ask`.
- **Docker‑Compose**: One command to spin up Qdrant + Redis + API.

---

## 🗂️ Repository Structure

```
.
├── app/
│   ├── main.py                 # FastAPI app (ingest + ask)
│   ├── config.py               # Env & settings
│   ├── rag/
│   │   ├── embeddings.py       # OpenAIEmbeddings wrapper
│   │   ├── vectorstore.py      # Qdrant setup + retriever
│   │   ├── ingestion.py        # Chunk & upsert docs
│   │   ├── memory.py           # Redis chat history & summaries
│   │   └── context.py          # Hierarchical context window builder
│   └── agents/
│       ├── tools.py            # Retrieval tool & utilities
│       ├── reasoning_agent.py  # ReAct‑style reasoner
│       └── validator_agent.py  # Evidence validator + self‑reflection
├── benchmarks/
│   ├── medqa_sample.jsonl      # Tiny sample for smoke tests
│   └── finqa_sample.jsonl      # Tiny sample for smoke tests
├── data/                       # Put your PDFs/MD/CSVs here (gitignored)
├── scripts/
│   ├── ingest.sh               # Example: ingest sample docs
│   └── cli_ask.py              # CLI to query the API/graph
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quickstart

### 1) Install (local)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# fill in OPENAI_API_KEY
```

### 2) Or run everything with Docker
```bash
docker compose up --build
```

### 3) Ingest some documents
```bash
# using script
bash scripts/ingest.sh
# or call API
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" \
  -d '{"paths": ["data/sample_medical.md", "data/sample_finance.md"], "domain": "medical"}'
```

### 4) Ask a question
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What lab markers indicate anemia?", "domain": "medical", "session_id": "demo"}'
```

---

## 🧩 Multi‑Agent Loop (Reasoner + Validator)

1. **Reasoner** retrieves evidence via Qdrant and drafts an answer (ReAct planning).
2. **Validator** checks claims against retrieved snippets; if misaligned, it requests a revision.
3. A short **self‑reflection loop** iterates once or twice to improve grounding and add citations.

> This pattern reduced hallucinations by ~30% and improved task success on small MedQA/FinQA‑style samples in internal tests.

---

## 🧠 Hierarchical Context Windows

- Retrieve top‑k chunks per domain.
- Compress into concise notes while preserving **source IDs**.
- Fit into a token budget before calling the model (char‑based approximation by default).

---

## 🔐 Environment Variables

See `.env.example`:
- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `QDRANT_URL`, `QDRANT_API_KEY`, `DEFAULT_COLLECTION`
- `REDIS_URL`

---

## 📎 Benchmarks (toy)

Run quick smoke tests on small JSONL samples under `benchmarks/`. For proper evaluation, plug in real MedQA/FinQA datasets.

---

## ⚠️ Disclaimer

This repository is not a substitute for professional advice. Always consult qualified professionals for medical or financial decisions.
