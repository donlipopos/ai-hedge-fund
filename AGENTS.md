# AI Hedge Fund - Project Context

## Project Overview
An AI-powered hedge fund proof-of-concept that leverages a multi-agent architecture to simulate trading decisions. The system uses **LangGraph** to coordinate specialized agents (inspired by famous investors like Buffett, Graham, and Lynch) alongside technical, fundamental, and sentiment analysts.

### Core Technology Stack
- **Languages:** Python (3.11+), TypeScript (React/Vite)
- **AI Orchestration:** LangGraph, LangChain
- **LLM Support:** Multi-provider (OpenAI, Anthropic, DeepSeek, Google Gemini, Groq, Kimi, MiniMax, Ollama, etc.)
- **API Framework:** FastAPI (Backend)
- **Data Sourcing:** `financial-datasets.com`, `Miaoxiang (MX)` API for Chinese A-shares
- **Development Tools:** Poetry, Docker, Pytest

### Architecture Highlights
- **Graph-Based Workflow:** Defined in `src/graph/state.py` and implemented in `src/main.py`. The graph parallelizes analysts, aggregates signals via a Risk Manager, and executes via a Portfolio Manager.
- **Agent Layer:** Located in `src/agents/`. Each agent is a functional node in the graph that uses LLMs and specialized tools to produce trading signals.
- **Data Layer:** Centralized in `src/tools/api.py` and `src/data/`. Features a caching system (`src/data/cache.py`) and an experimental v2 client layer (`v2/data/`).
- **Web App:** Full-stack implementation in `app/` with a React frontend and FastAPI backend.

---

## Building and Running

### Prerequisites
- Install **Poetry** for dependency management.
- Copy `.env.example` to `.env` and configure your API keys.

### CLI Commands
- **Install Dependencies:** `poetry install`
- **Run Hedge Fund:** `poetry run fund run --ticker AAPL,MSFT,NVDA`
- **Run Backtester:** `poetry run fund backtest --tickers AAPL,MSFT,NVDA`
- **A-Share Pipeline:** `poetry run fund ashare --criteria "ROE>10%"`

### Common CLI Parameters
All subcommands (`run`, `backtest`, `ashare`) support the following parameters:
- `--analysts`: Comma-separated analyst keys (e.g., `warren_buffett,technicals`).
- `--model`: Specific LLM model name (e.g., `gpt-4o`, `claude-3-5-sonnet-latest`).
- `--model-provider`: LLM provider (e.g., `OpenAI`, `Anthropic`, `Google`, `DeepSeek`).
- `--ollama`: Flag to use local Ollama models (interactive selection if no model specified).
- `--json`: (Global flag) Output raw JSON to `stdout` and silence progress bars/formatting.

### Web Application
- **Backend:** `cd app/backend && poetry run uvicorn main:app --reload`
- **Frontend:** `cd app/frontend && npm install && npm run dev`

### Testing
- **Run All Tests:** `poetry run pytest`
- **Specific Scopes:** `poetry run pytest tests/backtesting`

---

## Development Conventions

### Python Style
- **Indentation:** 4 spaces.
- **Formatters:** `black` (line-length 420), `isort`.
- **Naming:** `snake_case` for files and functions, `PascalCase` for classes.
- **Linting:** Use `flake8`.

### Frontend Style
- **Language:** TypeScript.
- **Indentation:** 2 spaces.
- **Naming:** `PascalCase` for components.

### Git & Pull Requests
- **Commit Messages:** Follow **Conventional Commits** (e.g., `feat(agent): ...`, `fix(llm): ...`).
- **PRs:** Keep changes focused and include validation evidence.

### Tooling Mandate
- **Context7:** Always use the `context7-mcp` skill to fetch current documentation for any library, framework, or cloud service before implementation.

---

## Key Directories
- `src/agents/`: Individual investor and analyst agent implementations.
- `src/tools/`: API adapters and data retrieval logic.
- `src/backtesting/`: Core engine for historical strategy simulation.
- `app/`: Web interface (FastAPI + React).
- `v2/`: Experimental next-generation data and feature engineering layer.
- `skills/`: Custom agent skills (e.g., MX data retrieval).
- `tests/`: Comprehensive test suites.

---

## Git Guidelines 
- Create feature branch for new features, refinements and enhancement before any code change. 
- Create hotfix branch for bugfix and hotfix before any code change. 
- Since this repository is a fork, the purpose of the main branch is to keep sync with the orignal main, never merge back to main. 
- Use the local branch for customized changes.
