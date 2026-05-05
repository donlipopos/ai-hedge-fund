---
name: ai-hedge-fund
display_name: AI Hedge Fund CLI
title: AI Hedge Fund skill
description: Use when running the AI hedge fund analysis pipeline, backtesting, or A-share screening via the fund CLI. Covers fund run, backtest, and ashare commands with project conventions.
homepage: https://github.com/don过往/ai-hedge-fund
author: Don
version: 1.0.0
metadata:
  hermes:
    tags: [hedge-fund, langgraph, trading, backtesting, a-share, cli]
    related_skills: [mx-data, mx-xuangu]
---

# AI Hedge Fund CLI

## Overview

An AI-powered hedge fund proof-of-concept using LangGraph multi-agent architecture to simulate trading decisions. Coordinates specialized agents (Buffett, Graham, Lynch styles) with technical, fundamental, and sentiment analysts.

Project root: `/Users/donli/Workspaces/ai-hedge-fund`

## When to Use

- Running stock analysis on US/A-share tickers
- Backtesting trading strategies
- Screening A-share stocks with financial criteria
- Inspecting agent graph architecture
- Building on the LangGraph workflow

## CLI Commands

### 1. Run hedge fund analysis
```bash
cd /Users/donli/Workspaces/ai-hedge-fund
python -m src.cli.entrypoint run --ticker AAPL,MSFT,NVDA
```
Interactive (prompts for tickers):
```bash
python -m src.cli.entrypoint run
```

### 2. Run backtest
```bash
cd /Users/donli/Workspaces/ai-hedge-fund
python -m src.cli.entrypoint backtest --tickers NVDA --start-date 2024-01-01
```

### 3. Run A-share screening pipeline
```bash
cd /Users/donli/Workspaces/ai-hedge-fund
python -m src.cli.entrypoint ashare --criteria "ROE>10%"
```

### Global flags
- `--json` — machine-readable JSON output (silences progress bars)

## Architecture

```
src/graph/state.py       — AgentState definition
src/main.py              — LangGraph workflow entry
src/agents/              — Investor & analyst agents
src/tools/api.py         — API adapters & data layer
src/data/cache.py        — Caching system
v2/data/                 — Experimental v2 data layer
app/                     — FastAPI + React web app
skills/                   — Custom MX skills (mx-data, mx-xuangu)
```

## Key Agents

| Agent | Style | Role |
|-------|-------|------|
| `portfolio_manager` | Buffett | Final portfolio decisions |
| `risk_manager` | Graham | Risk oversight |
| Buffett/Graham/Lynch | Investor | Fundamental analysis |
| Technical/Fundamental/Sentiment | Analyst | Signal generation |

## Project Conventions

### Python
- 4 spaces indent, `black` (line-length 420), `isort`
- `snake_case` files/functions, `PascalCase` classes
- Lint with `flake8`

### Git
- Feature branch for features, hotfix branch for bugs
- Main branch = sync with upstream, never merge back
- Conventional commits: `feat(agent): ...`, `fix(llm): ...`

### Tooling
- Always use `context7-mcp` skill for library/framework docs before implementation
- MX skills already available in `skills/mx-data/` and `skills/mx-xuangu/`

## Environment Setup

```bash
cp .env.example .env
# Edit .env and add your API keys (OpenAI, Anthropic, DeepSeek, Gemini, etc.)
```

## Web App

```bash
# Backend
cd app/backend && poetry run uvicorn main:app --reload

# Frontend
cd app/frontend && npm install && npm run dev
```

## Testing

```bash
poetry run pytest
poetry run pytest tests/backtesting
```

Note: `poetry` may not be in PATH. Use the venv Python directly:
```bash
/Users/donli/Workspaces/ai-hedge-fund/.venv/bin/python -m src.cli.entrypoint run --ticker AAPL
```

## Common Issues

1. **No module named 'langchain...'** — Run `poetry install` first
2. **poetry: command not found** — Use `.venv/bin/python -m src.cli.entrypoint` directly
3. **API key errors** — Check `.env` has correct keys set
4. **.venv missing binaries** — The venv may be incomplete; try `python -m venv .venv && poetry install`
