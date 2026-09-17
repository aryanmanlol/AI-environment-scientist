# 🌿 EcoIntel AI

## AI Biodiversity Assessment & Decision Support System

EcoIntel AI is an AI-powered environmental scientist that analyzes ecosystem conditions, identifies biodiversity risks, retrieves scientific evidence, performs multi-variable environmental reasoning, and generates actionable biodiversity improvement recommendations.

> **This is NOT a chatbot.** It is a scientific assessment engine that behaves like an environmental scientist + research analyst + decision support system.

---

## 🏗️ Architecture

```
User Input (Natural Language or Structured JSON)
      │
      ▼
┌─────────────────────────────┐
│   Input Validation Layer    │  ← Parse & validate environmental data
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│   Missing Data Detector     │  ← Identify gaps, ask follow-up questions
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Environmental Analyzer     │  ← Multi-variable ecosystem analysis
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Scientific Knowledge       │  ← RAG: retrieve from ChromaDB
│  Retrieval (RAG)            │     BAAI/bge-small-en-v1.5 embeddings
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Multi-Metric Reasoning     │  ← Cross-variable scientific reasoning
│  Engine                     │     Root causes, risk drivers, impacts
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Recommendation Engine      │  ← Evidence-backed action items
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Assessment Report          │  ← 10-section scientific report
│  Generator                  │
└─────────────────────────────┘
```

---

## 🧠 LangGraph Workflow

The system is orchestrated by a 7-node LangGraph StateGraph:

```
START
  │
  ▼
[Node 1] Input Processing ──────────────────────►
  │                                                │
  ▼                                                │
[Node 2] Missing Data Detection                    │
  │                                                │
  ├── (>2 critical fields missing) ──► END         │
  │     (returns follow-up questions)              │
  │                                                │
  ▼                                                │
[Node 3] Environmental Analysis                    │
  │                                                │
  ▼                                                │
[Node 4] Knowledge Retrieval (RAG)                 │
  │                                                │
  ▼                                                │
[Node 5] Scientific Reasoning                      │
  │                                                │
  ▼                                                │
[Node 6] Recommendation Generation                 │
  │                                                │
  ▼                                                │
[Node 7] Assessment Report Creation ──────────► END
```

---

## 📚 RAG Pipeline

```
PDF Documents (/data)
      │
      ▼
┌───────────────────┐
│  PyPDF Loader     │  ← Load PDF documents
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Text Chunking    │  ← RecursiveCharacterTextSplitter
│  (1000 chars,     │     200 char overlap
│   200 overlap)    │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Embeddings       │  ← BAAI/bge-small-en-v1.5
│  (384 dim)        │     Normalized for cosine similarity
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  ChromaDB         │  ← Persistent vector storage
│  Vector Store     │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  Multi-Query      │  ← 3-5 targeted queries per analysis
│  Retriever        │     Deduplication of results
└───────────────────┘
```

**Default Knowledge Base:** If no PDFs are provided, the system loads a built-in scientific knowledge base covering:
- Soil health & SOC dynamics
- Rainfall patterns & ecosystem impacts
- Monoculture effects on biodiversity
- Agroforestry benefits (ICRAF studies)
- Pollinator conservation (IPBES findings)
- Deforestation & habitat fragmentation
- Cover crop science
- Climate change impacts (IPCC)
- Water quality & aquatic biodiversity
- Soil microbiome & plant health
- Urban ecology & green infrastructure

---

## 🔬 Multi-Metric Reasoning Engine

The **most critical component**. Variables are NEVER analyzed independently.

### Example Reasoning Chain:

```
Low Soil Carbon (0.3%)
  + Low Rainfall
  + Monoculture Land Use
  ─────────────────────
  → Reduced microbial activity (SOC-dependent)
  → Lower vegetation productivity (NPP decline)
  → Habitat simplification (monoculture eliminates niches)
  → Pollinator decline (no floral resources)
  → Biodiversity decline risk: HIGH
```

### Output Structure:
- **Root Causes** with contributing factors and severity
- **Risk Drivers** connecting environmental variables
- **Ecological Impacts** with reversibility assessment
- **Confidence Score** based on data completeness and evidence alignment

---

## 📊 Assessment Report Format

Every report contains **10 structured sections**:

| # | Section | Description |
|---|---------|-------------|
| 1 | Ecosystem Health Status | Overall health assessment |
| 2 | Risk Level | LOW / MEDIUM / HIGH / CRITICAL |
| 3 | Root Cause Analysis | Multi-factor causal analysis |
| 4 | Scientific Interpretation | Evidence-based interpretation |
| 5 | Recommendations | Prioritized, actionable items |
| 6 | Impacted Metrics | Status of all environmental metrics |
| 7 | Time Horizon | Short-term to long-term outlook |
| 8 | Confidence Score | 0-1 with justification |
| 9 | Scientific Evidence | Retrieved knowledge with sources |
| 10 | References | All cited sources |

Each **recommendation** includes:
- ✅ What to do
- 🔬 Why it works (scientific rationale)
- 📈 Which metrics improve
- 📊 Expected quantitative impact
- ⏰ Time horizon
- 📖 Supporting evidence

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Google AI API key ([Get one here](https://aistudio.google.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/aryanmanlol/AI-environment-scientist.git
cd AI-environment-scientist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Run the Server

```bash
python app.py
```

Or with Docker:
```bash
docker build -t ecointel-ai .
docker run -p 8000:8000 --env-file .env ecointel-ai
```

The API will be available at `http://localhost:8000`.

---

## 📡 API Documentation

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "EcoIntel AI",
  "version": "1.0.0",
  "knowledge_base_stats": {
    "collection_name": "ecointel_knowledge",
    "count": 24
  }
}
```

### `POST /analyze`

Analyze environmental data. Accepts natural language or structured JSON, and an
optional `session_id` for multi-turn conversations.

**First turn (natural language, incomplete):**
```json
{
  "natural_language": "My biodiversity is declining."
}
```

**Response — asks a clarifying question and returns a `session_id`:**
```json
{
  "status": "needs_more_info",
  "session_id": "a1b2c3d4-...",
  "questions": [
    "What is the approximate soil organic carbon percentage?",
    "What type of land use practice is followed?",
    "What has rainfall been like recently?"
  ],
  "missing_fields": ["soil_carbon", "land_use", "rainfall"]
}
```

**Second turn — pass the same `session_id` so the answer is merged with what was already collected, rather than starting over:**
```json
{
  "session_id": "a1b2c3d4-...",
  "natural_language": "Soil carbon is 0.3%, rainfall is low, we grow monoculture wheat in a semi-arid region."
}
```

**Structured Request (single turn, all fields known):**
```json
{
  "input_data": {
    "soil_carbon": 0.3,
    "rainfall": "low",
    "temperature": 35,
    "land_use": "monoculture",
    "region": "semi-arid",
    "species_richness": "low"
  }
}
```

**Response (Complete Analysis):**
```json
{
  "status": "completed",
  "session_id": "a1b2c3d4-...",
  "result": {
    "report_id": "uuid",
    "ecosystem_health_status": "...",
    "risk_level": "HIGH",
    "root_cause_analysis": [...],
    "recommendations": [...],
    "confidence_score": 0.85,
    "formatted_text": "... full text report ..."
  }
}
```

If `session_id` is omitted, each call is treated as an independent, single-turn assessment (no memory carried forward).

### `POST /assessment`

Same as `/analyze` but guarantees a structured `AssessmentReport` response.

---

## 📁 Project Structure

```
ecointel-ai/
├── app.py                      # FastAPI application entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── modules/
│   ├── __init__.py             # Module initialization
│   ├── models.py               # Pydantic data models & GraphState
│   ├── analyzer.py             # Environmental input analyzer (NL + structured)
│   ├── reasoner.py             # Multi-metric scientific reasoning engine
│   ├── retriever.py            # RAG: PDF loading, embeddings, ChromaDB
│   ├── recommendations.py     # Recommendation formatting & fallback rules
│   ├── report_generator.py    # Assessment report generation
│   └── workflow.py             # LangGraph workflow orchestrator
│
├── prompts/
│   ├── __init__.py             # Prompts package
│   └── templates.py            # All prompt templates
│
├── data/                       # Place PDF documents here for RAG
│   └── README.md               # Knowledge base documentation
│
└── vectordb/                   # ChromaDB persistent storage (auto-generated)
    └── .gitkeep
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI |
| Agent Framework | LangGraph + LangChain |
| LLM | Gemini 2.5 Flash |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Data Validation | Pydantic v2 |

---

## 🗄️ Database / Schema

The system has no relational database. Persistent state lives in two places:

**Vector store (ChromaDB, on disk at `vectordb/`)**
Single collection `ecointel_knowledge`. Each entry:

| field         | type            | notes                                                    |
|---------------|-----------------|-----------------------------------------------------------|
| id            | auto (Chroma)   |                                                             |
| page_content  | text            | scientific fact chunk (from PDFs in `data/`, or the built-in fallback knowledge set if `data/` has none) |
| embedding     | vector(384)     | BAAI/bge-small-en-v1.5, cosine-normalized                  |
| metadata.source | text          | originating report / institution                           |
| metadata.topic  | text          | e.g. "Soil Health", "Agroforestry"                          |
| metadata.year   | int           | publication year, where known                               |

`load_and_index_documents()` is idempotent: it checks the collection count before indexing, so restarting the app does not duplicate chunks. Delete `vectordb/` to force a clean re-index (e.g. after adding new PDFs to `data/`).

**Conversation state (LangGraph `MemorySaver`, in-process)**
Each `session_id` maps to a `GraphState` checkpoint (`parsed_input`, `missing_fields`, `iteration_count`, etc.). This is what lets a multi-turn conversation — e.g. "biodiversity is declining" → clarifying question → "soil carbon is 0.3%, monoculture wheat" — accumulate fields across calls instead of resetting each time. It is in-memory and per-process; swap `MemorySaver` for a persistent LangGraph checkpointer (e.g. Postgres-backed) before running multiple workers or restarting between a user's turns.

---

## 🔄 CI/CD

`.github/workflows/ci.yml` runs on every push and PR:
1. Installs dependencies from `requirements.txt`
2. Byte-compiles `app.py`, `cli.py`, `modules/`, `prompts/` as a fast smoke test
3. Runs `tests/test_components.py` with a dummy API key (tests cover Pydantic model validation and the rule-based recommendation fallback, so they don't require a real Gemini key)

A `Dockerfile` is included for containerized deployment:
```bash
docker build -t ecointel-ai .
docker run -p 8000:8000 --env-file .env ecointel-ai
```
There is no deployment step wired into CI yet — add one (Render, Fly.io, Cloud Run) targeting this Dockerfile for a live demo URL.

---

## 🔮 Future Improvements

- **Time-series analysis**: Track ecosystem changes over time with historical data
- **Satellite imagery integration**: Use remote sensing for land cover analysis
- **Species database**: IUCN Red List API integration for species-specific risk assessment
- **Multi-region comparison**: Compare ecosystem health across geographic regions
- **Interactive dashboards**: Visualization of environmental metrics and trends
- **Webhook notifications**: Alert stakeholders when risk levels change
- **PDF report export**: HTML export exists via `modules/pdf_exporter.py` (CLI only, see `cli.py --html`); extend to real PDF and wire it into the API
- **Multi-language support**: Support for regional languages in input/output
- **Federated knowledge**: Connect to multiple scientific databases (Scopus, PubMed)
- **Confidence calibration**: ML-based confidence scoring with validation datasets

---

