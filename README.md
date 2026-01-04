# Vidhi AI - नेपाली कानुनी सहायक

> An intelligent Nepali Legal Assistant powered by RAG (Retrieval-Augmented Generation) and LLMs

Vidhi AI is a production-ready AI assistant that helps users understand Nepali laws by answering questions in English, Nepali, and Roman Nepali. It uses vector search to retrieve relevant legal documents and provides accurate, context-aware answers with source citations.

---

## ✨ Features

- **Multi-language Support**: Understands English, Nepali (देवनागरी), and Roman Nepali
- **RAG-based Architecture**: Retrieves relevant legal documents before generating answers
- **Intent Detection**: Automatically identifies legal queries vs. greetings/off-topic questions
- **Source Citations**: Every answer includes references to actual legal documents
- **Conversation History**: Maintains context across multiple messages
- **User Sessions**: Multi-user support with isolated chat sessions
- **PDF Processing Pipeline**: Automatically extracts and processes legal PDFs
- **Vector Search**: Fast semantic search across thousands of legal documents
- **Confidence Scoring**: Indicates reliability of generated answers
- **Comprehensive Logging**: Step-by-step tracking for debugging and monitoring

---

## 🛠️ Tech Stack

### Backend
- **Language**: Python 3.12
- **Framework**: FastAPI (async web framework)
- **Database**: PostgreSQL (user data, chat history)
- **Vector Database**: Pinecone (semantic search)
- **ORM**: SQLAlchemy

### AI/ML

Used Factory method for all so switching is matters of second.

*Change any of these in .env and every model for chat, embedding, vector storage, pdf parsing can be changed with ease*

- **LLM Providers**: 
  - Groq (fast inference)
  - Google Gemini (multilingual support)
  - OpenAI, Anthropic (optional)
- **Embeddings**: Pinecone `multilingual-e5-large` (1024 dimensions) default
- **Orchestration**: LangChain
- **PDF Parsing**: LLama Parse, Tesseract OCR

### Frontend
- **HTML/CSS/JavaScript** (Vanilla)
- **Icons**: Lucide Icons
- **UI**: Responsive chat interface

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Environment Management**: python-decouple
- **Logging**: Custom centralized logger with color-coded console output

---

## 🏗️ Architecture

Vidhi AI follows a modular RAG (Retrieval-Augmented Generation) architecture:

1. **User Query** → FastAPI endpoint
2. **Intent Detection** → LLM classifies query type and language
3. **Vector Search** → Pinecone retrieves top-K relevant legal documents
4. **Context Injection** → Legal documents added to LLM prompt
5. **Answer Generation** → LLM generates answer with source citations
6. **Response** → Structured JSON with answer, sources, and confidence

**Key Design Patterns**:
- Factory pattern for LLM/Embedding/Vector store providers
- Dependency injection for service management
- Repository pattern for database operations
- Pipeline pattern for data ingestion

---

## 📁 Folder Structure

```
vidhi_ai/
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Multi-container setup
├── Dockerfile                   # Python app container
├── Makefile                     # Common commands
│
├── data/                        # Data storage
│   ├── raw/                     # Original PDF files
│   ├── processed/               # Extracted markdown
│   ├── structured/              # Chunked documents ready for ingestion
│   ├── metadata/                # Document metadata (JSON)
│   └── *.json                   # Processing trackers
│
├── frontend/                    # Web UI
│   ├── index.html               # Chat interface
│   ├── script.js                # Frontend logic
│   └── style.css                # Styling
│
├── src/                         # Application source code
│   ├── core/                    # Core business logic
│   │   ├── llm_factory.py       # LLM provider factory
│   │   ├── embed_factory.py     # Embedding model factory
│   │   ├── vector_factory.py    # Vector database factory
│   │   ├── llm_config.py        # Configuration loader
│   │   ├── prompt_factory.py    # Prompt templates
│   │   └── secrets.py           # Environment variable manager
│   │
│   ├── llm/                     # LLM service layer
│   │   ├── calls.py             # LLMService (RAG pipeline)
│   │   └── dependencies.py      # FastAPI dependencies
│   │
│   ├── database/                # Database layer
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── session.py           # Database connection
│   │   └── base.py              # Base model
│   │
│   ├── crud/                    # Database operations
│   │   ├── user_ops.py          # User CRUD
│   │   └── chat_ops.py          # Chat session CRUD
│   │
│   ├── routes/                  # API endpoints
│   │   ├── user.py              # /api/users
│   │   ├── chat.py              # /api/chat
│   │   └── index.py             # Router aggregator
│   │
│   ├── schemas/                 # Pydantic models (validation)
│   │   ├── user_schemas.py
│   │   └── chat_schemas.py
│   │
│   ├── ingestion/               # Data processing pipeline
│   │   ├── collector.py         # PDF collection
│   │   ├── parser_factory.py    # PDF → Markdown
│   │   ├── postprocessor.py     # Chunking & structuring
│   │   └── ingestion_pipeline.py # Vector DB upload
│   │
│   ├── prompts/                 # LLM prompt templates
│   │   ├── identify_intent.tmpl
│   │   └── answer_legal_question.tmpl
│   │
│   └── utils/                   # Utilities
│       └── logger.py            # Centralized logging
│
└── scripts/
    └── init.sql                 # Database initialization
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.12+**
- **PostgreSQL 14+**
- **Docker & Docker Compose** (optional, recommended)
- **API Keys** (at least one):
  - Pinecone API key (required for vector search)
  - Groq API key (recommended)
  - Google Gemini API key (optional)
  - OpenAI/Anthropic API key (optional)

### Step 1: Clone Repository

```bash
git clone https://github.com/prabigya-pathak108/vidhi-ai-work
cd vidhi_ai
```

### Step 2: Environment Variables

Create a `.env` file in the root directory:

```env
# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/vidhi_ai

# Vector Database (Required)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=vidhi-ai-legal-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Intent Detection LLM
INTENT_LLM_PROVIDER=groq
INTENT_LLM_MODEL=llama-3.3-70b-versatile
INTENT_LLM_TEMPERATURE=0.1
GROQ_API_KEY=your-groq-api-key

# Legal Answer LLM
LEGAL_LLM_PROVIDER=google
LEGAL_LLM_MODEL=gemini-2.0-flash-exp
LEGAL_LLM_TEMPERATURE=0.3
GEMINI_API_KEY=your-gemini-api-key

# Optional: OpenAI
OPENAI_API_KEY=your-openai-api-key

# Optional: Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key

# Embedding Model (Pinecone Inference)
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=multilingual-e5-large

# RAG Configuration
RAG_TOP_K=5

# Data Paths
RAW_DATA_PATH=data/raw
PROCESSED_DATA_PATH=data/processed
STRUCTURED_DATA_PATH=data/structured
METADATA_PATH=data/metadata
```

### Step 3: Install Dependencies

**Option A: Docker (Recommended)**

```bash
docker-compose up -d
```

This will:
- Start PostgreSQL container
- Build and run the FastAPI application
- Expose API on `http://localhost:8000`

**Option B: Local Installation**

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL (ensure it's running)
# Create database
createdb vidhi_ai

# Run migrations (tables are created automatically on startup)
python main.py
```

### Step 4: Initialize Vector Database

Create Pinecone index (if not exists):

```bash
# The application will automatically create the index on first run
# Or manually create via Pinecone console with:
# - Name: vidhi-ai-legal-index
# - Dimensions: 1024
# - Metric: cosine
```

### Step 5: Ingest Legal Documents

```bash
# Process PDFs and upload to vector database
python -m src.ingestion.ingestion_pipeline
```

---

## 📖 Usage

### Starting the Server

```bash
# With Docker
docker-compose up

# Without Docker
python main.py
```

Server will start on: `http://localhost:8000`

### API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

### Web Interface

Open browser and navigate to: `http://localhost:8000`

**Steps**:
1. Enter a User ID (any string)
2. Click "Start Chat"
3. Ask legal questions in English, Nepali, or Roman Nepali
4. View answers with source citations


---

## ⚙️ Configuration

### LLM Provider Configuration

You can use different LLM providers for intent detection and legal answers:

```env
# Fast intent detection with Groq
INTENT_LLM_PROVIDER=groq
INTENT_LLM_MODEL=llama-3.3-70b-versatile
INTENT_LLM_TEMPERATURE=0.1

# High-quality legal answers with Gemini
LEGAL_LLM_PROVIDER=google
LEGAL_LLM_MODEL=gemini-2.0-flash-exp
LEGAL_LLM_TEMPERATURE=0.3
```

**Supported Providers**:
- `groq` - Fast inference (Llama models)
- `google` / `gemini` - Multilingual support
- `openai` - GPT models
- `anthropic` - Claude models
- `huggingface` - Open-source models

### RAG Configuration

```env
# Number of documents to retrieve from vector database
RAG_TOP_K=5

# Higher = more context, slower response
# Lower = faster, less context
```

### Embedding Model

```env
EMBEDDING_PROVIDER=pinecone
EMBEDDING_MODEL=multilingual-e5-large  # 1024 dimensions

# Alternative:
# EMBEDDING_PROVIDER=openai
# EMBEDDING_MODEL=text-embedding-3-large
```

### Logging

Logs are written to `vidhi_ai.log` and console.

Log level can be changed in `main.py`:

```python
setup_application_logging(log_level="INFO", log_file="vidhi_ai.log")
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Data Processing Pipeline

```bash
# Step 1: Collect PDFs from source
python -m src.ingestion.collector

# Step 2: Extract text from PDFs
python -m src.ingestion.data_collection_processing_pipeline

# Step 3: Chunk and structure documents
python -m src.ingestion.postprocessor

# Step 4: Generate embeddings and upload to Pinecone
python -m src.ingestion.ingestion_pipeline
```

### Database Migrations

Tables are automatically created on first run. To reset:

```sql
-- Connect to PostgreSQL
psql -d vidhi_ai

-- Drop all tables
DROP TABLE chat_messages CASCADE;
DROP TABLE chat_sessions CASCADE;
DROP TABLE users CASCADE;
```

---

## 🧪 Testing

### Running Tests

Vidhi AI includes simple tests using pytest.

**Install pytest:**

```bash
pip install pytest
```

**Run all tests:**

```bash
pytest tests/ -v
```

**Or use Make:**

```bash
make test
```

### Test Structure

```
tests/
├── conftest.py              # Test configuration
├── test_core_factories.py   # Factory validation tests
├── test_database.py         # Database model tests
└── test_utils.py            # Utility tests
```

### What's Tested

- ✅ **Factory Validation** - Ensures factories reject invalid inputs
- ✅ **Database Models** - Verifies model creation works correctly
- ✅ **Logger Utility** - Tests logging functionality

### GitHub Actions

Tests run automatically on every push via GitHub Actions.

**CI Status:**
[![CI Tests](https://github.com/yourusername/vidhi_ai/workflows/CI%20Tests/badge.svg)](https://github.com/yourusername/vidhi_ai/actions)

### PostgreSQL for Tests

Tests use the same PostgreSQL database as development. Make sure PostgreSQL is running:

```bash
# Check PostgreSQL is running
docker-compose ps

# Or if running locally
systemctl status postgresql
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Your Name** - Prabigya Pathak

---

## 🙏 Acknowledgments

- Nepal Law Commission for legal documents
- LangChain for LLM orchestration
- Pinecone for vector search infrastructure
- FastAPI for the web framework

---



**Built with ❤️ for the Nepali legal community**
