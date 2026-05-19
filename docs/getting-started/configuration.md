# Configuration

All settings can be passed directly to `QueryEngine` or set via environment variables
(prefixed `AAIZAQL_`). Environment variables are read automatically — no extra setup needed.

## Environment variables

| Setting | Env var | Default | Description |
|---|---|---|---|
| LLM provider | `AAIZAQL_LLM_PROVIDER` | `groq` | `groq`, `claude`, `openai`, `ollama` |
| Groq API key | `AAIZAQL_GROQ_API_KEY` | — | Free key at console.groq.com |
| Groq model | `AAIZAQL_GROQ_MODEL` | `llama3-70b-8192` | Any Groq-supported model |
| Anthropic key | `AAIZAQL_ANTHROPIC_API_KEY` | — | For `llm="claude"` |
| OpenAI key | `AAIZAQL_OPENAI_API_KEY` | — | For `llm="openai"` |
| Ollama URL | `AAIZAQL_OLLAMA_BASE_URL` | `http://localhost:11434` | For local models |
| Vector store | `AAIZAQL_VECTOR_STORE` | `chroma` | `chroma` or `qdrant` |
| Chroma dir | `AAIZAQL_CHROMA_PERSIST_DIR` | `.AAIZAQL_chroma` | ChromaDB storage path |
| Max retries | `AAIZAQL_MAX_SELF_CORRECTION_RETRIES` | `3` | Self-correction attempts |
| Session history | `AAIZAQL_SESSION_HISTORY_LIMIT` | `10` | Turns kept in context |
| Injection detection | `AAIZAQL_ENABLE_INJECTION_DETECTION` | `true` | Prompt injection scanning |

## Using a `.env` file

Create a `.env` file in your project root — AAIZAQL loads it automatically:

```bash
AAIZAQL_GROQ_API_KEY=gsk_your_key_here
AAIZAQL_LLM_PROVIDER=groq
AAIZAQL_VECTOR_STORE=chroma
```

## Passing settings directly

```python
from aaizaql import QueryEngine

engine = QueryEngine(
    llm="claude",
    database="postgresql",
    dsn="postgresql://user:pass@localhost/mydb",
    max_self_correction_retries=5,
    session_history_limit=20,
)
```

## Using Qdrant instead of ChromaDB

For production deployments, Qdrant is recommended over ChromaDB:

```bash
pip install "AAIZAQL[qdrant]"
docker run -p 6333:6333 qdrant/qdrant
```

```bash
AAIZAQL_VECTOR_STORE=qdrant
AAIZAQL_QDRANT_URL=http://localhost:6333
```