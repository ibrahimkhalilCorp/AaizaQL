# Installation

## Requirements

- Python 3.11 or higher
- pip

## Basic install

```bash
pip install aaizaql
```

## Install with your LLM provider

aaizaql supports multiple LLM providers. Install the one you want to use:

```bash
# Groq — free tier, fastest inference, recommended for getting started
pip install "AAIZAQL[groq]"

# Anthropic Claude — best accuracy on complex schemas
pip install "AAIZAQL[claude]"

# OpenAI — GPT-4o and others
pip install "AAIZAQL[openai]"

# Ollama — local models, fully private, no API key needed
pip install AAIZAQL   # ollama provider uses requests, already included
```

## Install with your database driver

```bash
# PostgreSQL
pip install "AAIZAQL[postgres]"

# MySQL
pip install "AAIZAQL[mysql]"

# Snowflake
pip install "AAIZAQL[snowflake]"

# DuckDB
pip install "AAIZAQL[duckdb]"

# SQLite is built into Python — no extra install needed
```

## Install everything

```bash
pip install "AAIZAQL[all]"
```

## Development install

```bash
git clone https://github.com/ibrahimkhalil/AAIZAQL
cd AAIZAQL
pip install -e ".[dev]"
pre-commit install
```

## Verify the install

```bash
AAIZAQL version
```

You should see `AAIZAQL 0.1.0`.