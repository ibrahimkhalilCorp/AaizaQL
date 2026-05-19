# Installation

## Requirements

- Python 3.11 or higher
- pip

## Basic install

```bash
pip install aqlix
```

## Install with your LLM provider

aqlix supports multiple LLM providers. Install the one you want to use:

```bash
# Groq — free tier, fastest inference, recommended for getting started
pip install "aqlix[groq]"

# Anthropic Claude — best accuracy on complex schemas
pip install "aqlix[claude]"

# OpenAI — GPT-4o and others
pip install "aqlix[openai]"

# Ollama — local models, fully private, no API key needed
pip install aqlix   # ollama provider uses requests, already included
```

## Install with your database driver

```bash
# PostgreSQL
pip install "aqlix[postgres]"

# MySQL
pip install "aqlix[mysql]"

# Snowflake
pip install "aqlix[snowflake]"

# DuckDB
pip install "aqlix[duckdb]"

# SQLite is built into Python — no extra install needed
```

## Install everything

```bash
pip install "aqlix[all]"
```

## Development install

```bash
git clone https://github.com/ibrahimkhalil/aqlix
cd aqlix
pip install -e ".[dev]"
pre-commit install
```

## Verify the install

```bash
aqlix version
```

You should see `aqlix 0.1.0`.