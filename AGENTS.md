# Repository Guidelines

## Project Structure & Module Organization

Core Python trading logic lives in `src/`, with agents in `src/agents/`, CLI entrypoints in `src/main.py` and `src/backtester.py`, shared utilities in `src/utils/`, and backtesting code in `src/backtesting/`. Experimental next-generation work is isolated in `v2/`. The web app lives under `app/`: `app/backend/` contains the FastAPI server, database models, Alembic migrations, and service layer; `app/frontend/` contains the Vite/React UI. Tests live in `tests/`, with backtesting coverage under `tests/backtesting/` and reusable API fixtures under `tests/fixtures/`.

## Build, Test, and Development Commands

Install Python dependencies from the repository root with `poetry install`. Run the CLI hedge fund with `poetry run python src/main.py --ticker AAPL,MSFT,NVDA`. Run the backtester with `poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA` or `poetry run backtester`. Start the API locally from `app/backend/` with `poetry run uvicorn main:app --reload`. Start the frontend from `app/frontend/` with `npm install` and `npm run dev`. Production frontend builds use `npm run build`; lint with `npm run lint`.

## Coding Style & Naming Conventions

Python targets 3.11 and uses `black`, `isort`, and `flake8`; format before opening a PR with `poetry run black . && poetry run isort . && poetry run flake8`. Use 4-space indentation, `snake_case` for modules/functions, and `PascalCase` for classes. Frontend code is TypeScript + React with ESLint; existing files use 2-space indentation, `PascalCase` component names, and `kebab-case` filenames only for utility-style assets when already established.

## Testing Guidelines

Use `pytest` from the repo root: `poetry run pytest`. Narrow scope during development, for example `poetry run pytest tests/backtesting -q` or `poetry run pytest tests/test_cache.py`. Name test files `test_*.py`, group related assertions in `Test...` classes when helpful, and keep deterministic fixtures in `tests/fixtures/` so API-dependent logic remains reproducible.

## Commit & Pull Request Guidelines

Recent history favors short, imperative Conventional Commit subjects such as `feat(data): add MX A-share data layer` and `feat(llm): add MiniMax M2.7 as LLM provider`. Follow that pattern when possible. Keep PRs focused, explain behavior changes, list validation steps, and attach screenshots for UI work. Link the relevant issue when one exists.

## Configuration & Agent Notes

Copy `.env.example` to `.env` and supply required API keys before running the CLI or app. Do not commit secrets. When updating library, framework, SDK, API, CLI, or cloud-service usage, fetch current docs through Context7 before coding or revising examples.
