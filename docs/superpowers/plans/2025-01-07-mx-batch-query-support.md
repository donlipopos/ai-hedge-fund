# MX API Batch Query Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement batch querying for the Chinese A-Share pipeline to reduce remote API calls without modifying individual agent logic.

**Architecture:** Instead of refactoring 15+ individual analyst agents to support batch processing, we will implement a "Cache Warming" strategy. We will add `warm_*_cache(tickers)` methods to `src/tools/mx_adapter.py`. These methods will combine multiple tickers into a single natural language query (e.g., `"600519和000001..."`), parse the batch response, and populate the local cache using the exact `cache_key` formats expected by the single-ticker `get_*` functions. Finally, `ashare_pipeline.py` will call these warming methods before starting the LangGraph workflow. When agents run their loops, `get_prices` and `get_financial_metrics` will instantly hit the local cache, reducing network calls to near zero.

**Tech Stack:** Python, MX API, existing `src.data.cache`

---

### Task 1: API Spike - Investigate Batch Response Format

Before writing the caching logic, we need to understand how the Miaoxiang (MX) API formats tables when queried with multiple tickers (e.g., does it use a `代码` or `证券代码` column to distinguish rows?).

**Files:**
- Create: `scripts/spike_mx_batch.py`

- [ ] **Step 1: Write spike script**

```python
# scripts/spike_mx_batch.py
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from src.tools.mx_adapter import _mx_query_tables

def main():
    print("Testing Market Cap Batch...")
    tables, title, _, err = _mx_query_tables("600519和000001最新总市值流通市值")
    print(f"Error: {err}")
    if tables:
        print("Keys:", list(tables[0][0].keys()))
        print("Row 0:", tables[0][0])
        
    print("\nTesting Financial Metrics Batch...")
    tables, title, _, err = _mx_query_tables("600519和000001近1年的净利润、营业收入")
    print(f"Error: {err}")
    if tables:
        print("Keys:", list(tables[0][0].keys()))
        print("Row 0:", tables[0][0])

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run spike to observe output**

Run: `export MX_SKILL_PARENT=$(pwd)/skills && poetry run python scripts/spike_mx_batch.py`
*(Note the column name used for stock codes: e.g., '证券代码' or '代码'. You will need this for the next tasks.)*

- [ ] **Step 3: Commit (or delete the spike)**
```bash
rm scripts/spike_mx_batch.py
```

### Task 2: Add Ticker Chunking and Code Mapping Helpers

**Files:**
- Modify: `src/tools/mx_adapter.py`

- [ ] **Step 1: Implement `_chunk_tickers` and mapping logic**

```python
# Add to helpers section in mx_adapter.py
def _chunk_tickers(tickers: list[str], chunk_size: int = 5) -> list[list[str]]:
    return [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]

def _find_code_in_row(row: dict) -> str | None:
    """Extract stock code from a result row using common MX API column names."""
    for key in ["证券代码", "代码", "股票代码", "code"]:
        if key in row:
            # MX often returns codes with suffixes (e.g. 600519.SH) or prefixes
            val = str(row[key])
            # Strip suffixes/prefixes to get the 6-digit base code
            match = re.search(r"(\d{6})", val)
            if match:
                return match.group(1)
    return None
```

- [ ] **Step 2: Commit**

```bash
git add src/tools/mx_adapter.py
git commit -m "feat(data): add ticker chunking and code extraction helpers for MX batching"
```

### Task 3: Implement `warm_market_cap_cache`

**Files:**
- Modify: `src/tools/mx_adapter.py`

- [ ] **Step 1: Write `warm_market_cap_cache` method**

```python
def warm_market_cap_cache(tickers: list[str], end_date: str) -> None:
    """Pre-fetch and cache market cap for multiple A-share tickers."""
    cache = get_cache()
    if not hasattr(cache, '_market_cap_cache'):
        cache._market_cap_cache = {}
        
    ashare_tickers = [t for t in tickers if _is_ashare(t)]
    chunks = _chunk_tickers(ashare_tickers, chunk_size=5)
    
    for chunk in chunks:
        needed_tickers = []
        needed_codes = []
        for t in chunk:
            cache_key = f"{t}_{end_date}"
            if cache._market_cap_cache.get(cache_key) is None:
                needed_tickers.append(t)
                needed_codes.append(_ticker_to_code(t))
                
        if not needed_codes:
            continue
            
        query = f"{'和'.join(needed_codes)}最新总市值流通市值"
        tables, _, _, err = _mx_query_tables(query)
        if err or not tables:
            continue
            
        # Map codes back to full tickers (e.g. 600519 -> 600519.SH)
        code_to_ticker = {c: t for c, t in zip(needed_codes, needed_tickers)}
        
        for row in tables[0]:
            code = _find_code_in_row(row)
            if code and code in code_to_ticker:
                # Find the cap column
                cap_val = None
                for k, v in row.items():
                    if "市值" in k and "流通" not in k:
                        cap_val = _parse_chinese_number(str(v))
                        break
                
                if cap_val is not None:
                    ticker = code_to_ticker[code]
                    cache_key = f"{ticker}_{end_date}"
                    cache._market_cap_cache[cache_key] = cap_val
```

- [ ] **Step 2: Commit**
```bash
git add src/tools/mx_adapter.py
git commit -m "feat(data): implement market cap cache warming for MX A-shares"
```

### Task 4: Implement `warm_financial_metrics_cache`

**Files:**
- Modify: `src/tools/mx_adapter.py`

- [ ] **Step 1: Write `warm_financial_metrics_cache` method**

```python
def warm_financial_metrics_cache(tickers: list[str], end_date: str, period: str = "ttm", limit: int = 10) -> None:
    """Pre-fetch and cache financial metrics for multiple A-share tickers."""
    cache = get_cache()
    ashare_tickers = [t for t in tickers if _is_ashare(t)]
    chunks = _chunk_tickers(ashare_tickers, chunk_size=3) # Smaller chunks for dense metrics
    
    for chunk in chunks:
        needed_tickers = []
        needed_codes = []
        for t in chunk:
            cache_key = f"{t}_{period}_{end_date}_{limit}"
            if not cache.get_financial_metrics(cache_key):
                needed_tickers.append(t)
                needed_codes.append(_ticker_to_code(t))
                
        if not needed_codes:
            continue
            
        # We query just the most recent N periods
        query = (
            f"{'和'.join(needed_codes)}近{limit}年的"
            f"总市值、市盈率TTM、市净率、净资产收益率、毛利率、"
            f"营业收入同比、净利润同比、经营现金流、资产负债率、"
            f"流动比率、股息率、营业收入、净利润、"
            f"折旧与摊销、资本开支、总资产、总负债、净资产"
        )
        tables, _, _, err = _mx_query_tables(query)
        if err or not tables:
            continue
            
        code_to_ticker = {c: t for c, t in zip(needed_codes, needed_tickers)}
        
        # Group rows by code
        ticker_metrics = {t: [] for t in needed_tickers}
        for row in tables[0]:
            code = _find_code_in_row(row)
            if code and code in code_to_ticker:
                ticker = code_to_ticker[code]
                m = _parse_financial_metrics_row(row)
                if m.report_period:
                    ticker_metrics[ticker].append(m)
                    
        # Sort and cache
        for ticker, metrics in ticker_metrics.items():
            if metrics:
                # Sort descending by date
                metrics.sort(key=lambda x: x.report_period, reverse=True)
                result = metrics[:limit]
                cache_key = f"{ticker}_{period}_{end_date}_{limit}"
                cache.set_financial_metrics(cache_key, [m.model_dump() for m in result])
```

- [ ] **Step 2: Commit**
```bash
git add src/tools/mx_adapter.py
git commit -m "feat(data): implement financial metrics cache warming for MX A-shares"
```

### Task 5: Integrate Cache Warming into the Pipeline

**Files:**
- Modify: `src/cli/ashare_pipeline.py`

- [ ] **Step 1: Call warming methods before graph execution**

Modify `src/cli/ashare_pipeline.py` inside the `main()` function, right before `workflow.compile().invoke(...)`:

```python
from src.tools.mx_adapter import warm_market_cap_cache, warm_financial_metrics_cache

# ... inside main(), after gathering candidates ...
tickers = list(candidates.keys())

print(f"\n[System] Warming cache for {len(tickers)} tickers via batch queries to MX API...")
try:
    warm_market_cap_cache(tickers, end_date)
    warm_financial_metrics_cache(tickers, end_date, period="ttm", limit=10)
    print("[System] Cache warming complete.")
except Exception as e:
    print(f"[Warning] Cache warming failed: {e}. Falling back to single queries.")

# Build workflow
workflow = create_workflow(selected_analysts)
# ...
```

- [ ] **Step 2: Commit**
```bash
git add src/cli/ashare_pipeline.py
git commit -m "feat(cli): integrate batch cache warming into A-share pipeline"
```

### Task 6: End-to-End Test

- [ ] **Step 1: Run the pipeline to verify batching**

Run: 
```bash
export MX_SKILL_PARENT=$(pwd)/skills
poetry run python -m src.cli.ashare_pipeline \
    --criteria "市值大于2000亿的酿酒行业A股" \
    --max-candidates 3 \
    --end-date 2026-04-15 \
    --analysts warren_buffett \
    --model MiniMax-M2.7
```

Observe the output to verify:
1. The "Warming cache..." print statement appears.
2. The pipeline executes significantly faster during the `warren_buffett_agent` phase because the network calls are hitting the cache.

---
*Note: Due to the variability of NLP interpretation by the Miaoxiang API, batch extraction relies heavily on the `_find_code_in_row` heuristic working consistently across different query formulations.*