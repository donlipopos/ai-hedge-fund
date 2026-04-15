#!/usr/bin/env python3
"""
A-Share Pipeline — MiniMax-powered hedge fund analysis for Chinese A-shares.

Usage:
    MX_SKILL_PARENT=/path/to/skills \
    python -m src.cli.ashare_pipeline \\
        --criteria "ROE大于10%流通市值大于100亿的A股" \\
        --max-candidates 10 \\
        --end-date 2026-04-15 \\
        --analysts warren_buffett,valuation_analyst \\
        --model MiniMax-M2.7

Requirements:
    MX_APIKEY       (from mx-data skill, also in ~/.openclaw/.env)
    MINIMAX_API_KEY (from MiniMax platform, also in ~/.openclaw/.env)
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

import importlib.util

from src.llm.models import find_model_by_name
from src.main import run_hedge_fund
from src.cli.input import add_common_args, add_date_args


def _import_skill_module(skill_name: str, module_name: str):
    """Import a skill's Python module by name using file-based import (avoids hyphen-issue)."""
    skill_parent = os.path.expandvars(os.path.expanduser(
        os.getenv("MX_SKILL_PARENT", os.path.expanduser("~/.openclaw/workspace-trading/skills"))
    ))
    module_path = os.path.join(skill_parent, skill_name, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _code_to_ticker(code: str, market: str) -> str:
    """Convert mx-xuangu output to standard ticker format."""
    mkt = market.strip().upper()
    if mkt not in ("SH", "SZ", "BJ"):
        mkt = "SZ"
    return f"{code.strip()}.{mkt}"


def _run_xuangu_screening(criteria: str, max_candidates: int = 20) -> list[dict]:
    """
    Run MXSelectStock screening and return list of candidate dicts.
    """
    mx_xuangu = _import_skill_module("mx-xuangu", "mx_xuangu")
    MXSelectStock = mx_xuangu.MXSelectStock

    mx = MXSelectStock()
    result = mx.search(criteria)
    rows, _, err = MXSelectStock.extract_data(result)

    if err:
        print(f"[XUANGU] Screening error: {err}")
        return []

    today = datetime.now().strftime("%Y.%m.%d")
    candidates = []
    for row in rows[:max_candidates]:
        code = row.get("代码", "").strip()
        market = row.get("市场代码简称", "SZ").strip()
        name = row.get("名称", "").strip()
        if not code:
            continue
        candidates.append({
            "ticker": _code_to_ticker(code, market),
            "name": name,
            "code": code,
            "market": market,
            "latest_price": row.get(f"最新价(元) {today}", ""),
            "pct_change": row.get(f"涨跌幅(%) {today}", ""),
            "roe": row.get(f"净资产收益率ROE(加权)(%) 截至{today}最新", ""),
            "mkt_cap": row.get(f"流通市值(元) {today}", ""),
        })
    return candidates


def run_ashare_pipeline(
    criteria: str,
    max_candidates: int = 20,
    start_date: str | None = None,
    end_date: str | None = None,
    selected_analysts: list[str] | None = None,
    model_name: str = "MiniMax-M2.7",
    model_provider: str = "MiniMax",
    show_reasoning: bool = False,
    initial_cash: float = 1_000_000.0,
) -> dict:
    """
    Full A-share pipeline:
    1. Screen candidates via mx-xuangu
    2. Run agent pipeline on each
    3. Return aggregated signals
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    start_date = start_date or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    selected_analysts = selected_analysts or ["warren_buffett", "valuation_analyst"]

    # ── Step 1: Screen ──────────────────────────────────────────────
    print(f"\n🔍 Screening A-shares: {criteria}")
    candidates = _run_xuangu_screening(criteria, max_candidates)
    if not candidates:
        print("⚠️  No candidates found. Check your screening criteria.")
        return {"error": "No candidates", "signals": {}}

    tickers = [c["ticker"] for c in candidates]
    preview = ", ".join(f"{c['name']}({c['ticker']})" for c in candidates[:5])
    more = " ..." if len(candidates) > 5 else ""
    print(f"✅ {len(candidates)} candidates: {preview}{more}")

    # ── Step 2: Build portfolio ──────────────────────────────────────
    portfolio = {
        "cash": initial_cash,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0, "short": 0,
                "long_cost_basis": 0.0, "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {"long": 0.0, "short": 0.0}
            for ticker in tickers
        },
    }

    # ── Step 3: Run agent pipeline ─────────────────────────────────
    print(f"\n🤖 Running [{', '.join(selected_analysts)}] with {model_name} ...")
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=show_reasoning,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )

    # ── Step 4: Build enriched signal summary ───────────────────────
    decisions = result.get("decisions") or {}
    analyst_signals = result.get("analyst_signals") or {}

    summary = []
    for c in candidates:
        ticker = c["ticker"]
        decision = decisions.get(ticker, {})
        signals_for_ticker = {}
        for agent_id, signals in analyst_signals.items():
            if ticker in signals:
                signals_for_ticker[agent_id] = signals[ticker]

        summary.append({
            "ticker": ticker,
            "name": c["name"],
            "price": c["latest_price"],
            "pct_change": c["pct_change"],
            "roe": c["roe"],
            "mkt_cap": c["mkt_cap"],
            "action": decision.get("action", "unknown"),
            "quantity": decision.get("quantity", 0),
            "confidence": decision.get("confidence", 0),
            "reasoning": decision.get("reasoning", ""),
            "agent_signals": signals_for_ticker,
        })

    return {
        "criteria": criteria,
        "candidates": candidates,
        "tickers": tickers,
        "result": result,
        "summary": summary,
    }


def print_summary(pipeline_result: dict):
    """Pretty-print signal summary to console."""
    summary = pipeline_result.get("summary", [])
    if not summary:
        print("\n⚠️  No signals generated.")
        return

    SEP = "─" * 85
    print(f"\n{SEP}")
    print(f"📊 A-Share Signal Summary — {pipeline_result.get('criteria', '')}")
    print(SEP)

    hdr = (f"{'Code':<10} {'Name':<10} {'Price':>8} {'%Chg':>7} {'ROE':>8} "
           f"{'MktCap':>10} {'Action':<6} {'Qty':>6} {'Conf':>5}  Reasoning")
    print(hdr)
    print(SEP)

    for s in summary:
        pct = s["pct_change"] or "—"
        roe = s["roe"] or "—"
        mkt = s["mkt_cap"] or "—"
        action = s["action"] or "—"
        qty = s["quantity"] or 0
        conf = s["confidence"] or 0
        reason = (s["reasoning"] or "—")[:55]
        name = (s["name"] or "")[:10]

        print(
            f"{s['ticker']:<10} {name:<10} {str(s['price'] or '—'):>8} {str(pct):>7} "
            f"{str(roe):>8} {str(mkt):>10} {action:<6} {qty:>6} {conf:>5}  {reason}"
        )

    actions: dict[str, int] = {}
    for s in summary:
        a = s["action"] or "unknown"
        actions[a] = actions.get(a, 0) + 1

    print(SEP)
    print(f"Actions: {actions}")
    print(SEP + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="A-Share AI Hedge Fund Pipeline (MiniMax + 妙想 MX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--criteria", type=str, required=True,
        help="Natural-language screening criteria for mx-xuangu "
             "(e.g. 'ROE大于10%流通市值大于100亿的A股')",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=10,
        help="Maximum candidates to analyze (default: 10)",
    )
    add_date_args(parser, default_months_back=1)
    parser.add_argument(
        "--initial-cash", type=float, default=1_000_000.0,
        help="Initial portfolio cash in CNY (default: 1,000,000)",
    )
    parser.add_argument(
        "--analysts", type=str, default="warren_buffett,valuation_analyst",
        help="Comma-separated analyst keys (default: warren_buffett,valuation_analyst)",
    )
    parser.add_argument(
        "--model", type=str, default="MiniMax-M2.7",
        help="LLM model (default: MiniMax-M2.7)",
    )
    parser.add_argument(
        "--model-provider", type=str, default="MiniMax",
        help="LLM provider (default: MiniMax)",
    )
    parser.add_argument("--show-reasoning", action="store_true",
                        help="Print agent reasoning trace")
    parser.add_argument(
        "--output-json", type=str,
        help="Write summary JSON to this path",
    )
    add_common_args(parser, require_tickers=False, include_analyst_flags=False, include_ollama=False)

    args = parser.parse_args()
    selected_analysts = [a.strip() for a in args.analysts.split(",") if a.strip()]

    warnings.filterwarnings("ignore")

    result = run_ashare_pipeline(
        criteria=args.criteria,
        max_candidates=args.max_candidates,
        start_date=args.start_date,
        end_date=args.end_date,
        selected_analysts=selected_analysts,
        model_name=args.model,
        model_provider=args.model_provider,
        show_reasoning=args.show_reasoning,
        initial_cash=args.initial_cash,
    )

    print_summary(result)

    if args.output_json:
        import json
        serializable = {
            k: v for k, v in result.items()
            if k in ("criteria", "candidates", "tickers", "summary")
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 JSON output written to: {args.output_json}")


if __name__ == "__main__":
    main()
