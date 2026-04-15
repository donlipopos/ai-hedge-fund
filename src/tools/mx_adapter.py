"""
MX Adapter — A-share data layer via 妙想 (MX) API.

Provides the same function signatures as src/tools/api.py so agents can fetch
A-share (沪深) data without any caller-side changes — just set
`USE_MX=true` or a ticker's market is detected as A-share.

Environment:
    MX_APIKEY          — MX API key (also loaded from .env)
    MX_OUTPUT_DIR      — MX script output dir (default: /tmp/mx_output)
"""

from __future__ import annotations

import os
import re
import sys
import json
import logging
from datetime import datetime
from typing import Optional

# Ensure the MX skill's mx_data.py is importable
_MX_SKILL_PATH = os.path.expanduser(
    os.getenv("MX_SKILL_PATH", "~/.openclaw/workspace-trading/skills/mx-data")
)
if _MX_SKILL_PATH not in sys.path:
    sys.path.insert(0, _MX_SKILL_PATH)

try:
    from mx_data import MXData
except ImportError as e:
    raise ImportError(
        f"mx_data module not found at {MX_SKILL_PATH}. "
        "Set MX_SKILL_PATH env var or ensure mx-data skill is installed."
    ) from e

from src.data.models import (
    Price,
    PriceResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    LineItem,
    LineItemResponse,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _get_mx_client() -> MXData:
    api_key = os.getenv("MX_APIKEY")
    if not api_key:
        raise ValueError("MX_APIKEY environment variable not set.")
    return MXData(api_key=api_key)


def _is_ashare(ticker: str) -> bool:
    """Detect A-share ticker (沪深)."""
    t = ticker.upper()
    return t.endswith(".SH") or t.endswith(".SZ") or t.endswith(".BJ")


def _ticker_to_code(ticker: str) -> str:
    """Strip market suffix: 300059.SZ → 300059."""
    return re.sub(r"\.(SH|SZ|BJ)$", "", ticker, flags=re.IGNORECASE)


def _parse_chinese_number(value: str) -> float:
    """
    Parse Chinese-formatted numbers with unit suffixes.

    Examples:
        "20.29元"    → 20.29
        "2.182亿股"  → 218200000.0
        "43.6亿元"   → 4360000000.0
        "38.46%"    → 0.3846
        "120.8亿元"  → 12080000000.0
        "3137亿"    → 313700000000.0
    """
    if not isinstance(value, str):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    s = value.strip()

    # Percentage
    if s.endswith("%"):
        try:
            return float(s[:-1].replace(",", "")) / 100.0
        except ValueError:
            return 0.0

    # Chinese compound units (must be checked before simple units)
    compound_units = {
        "亿元": 1e8,
        "万亿": 1e12,
        "兆元": 1e12,
        "亿股": 1e8,
        "万股": 1e4,
        "万份": 1e4,
        "亿份": 1e8,
    }
    for unit, multiplier in compound_units.items():
        if s.endswith(unit):
            num_part = s[: -len(unit)].replace(",", "").strip()
            try:
                return float(num_part) * multiplier
            except ValueError:
                return 0.0

    # Simple numeric suffix units
    simple_units = {
        "亿": 1e8,
        "万": 1e4,
        "兆": 1e12,
    }
    for unit, multiplier in simple_units.items():
        if s.endswith(unit):
            num_part = s[: -len(unit)].replace(",", "").strip()
            try:
                return float(num_part) * multiplier
            except ValueError:
                return 0.0

    # Strip trailing non-numeric characters and parse
    # e.g. "2.182亿股" → "2.182亿" → handled above
    # e.g. "3137亿" was handled
    # Strip trailing common non-numeric chars
    s = re.sub(r"[元股人份箱手]$", "", s)
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _clean_date(date_str: str) -> str:
    """
    Normalise MX date strings to YYYY-MM-DD.

    Examples:
        "2026-04-15(日)" → "2026-04-15"
        "2026-04-15"     → "2026-04-15"
        "2025年报"        → "2025-12-31"
    """
    s = date_str.strip()
    # Strip weekday annotation
    s = re.sub(r"[()]([^)]*)?$", "", s)  # "(日)" etc.
    # Try direct parse
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        pass
    # Annual report shorthand "2025年报" → last day of year
    m = re.match(r"^(\d{4})年报?$", s)
    if m:
        return f"{m.group(1)}-12-31"
    # Fallback: return as-is
    return s[:10] if len(s) >= 10 else s


# ─────────────────────────────────────────────────────────────────
# MX Query Wrappers
# ─────────────────────────────────────────────────────────────────

def _mx_query(query: str) -> dict:
    """
    Execute a natural-language MX query and return the raw JSON result.
    Caches per-query to avoid redundant API calls within the same session.
    """
    cache: dict[str, dict] = {}
    if query in cache:
        return cache[query]
    client = _get_mx_client()
    result = client.query(query)
    cache[query] = result
    return result


def _mx_query_tables(query: str) -> tuple[list, list[str], int, Optional[str]]:
    """Execute MX query and return parsed tables (same signature as MXData.parse_result)."""
    client = _get_mx_client()
    result = client.query(query)
    return MXData.parse_result(result)


# ─────────────────────────────────────────────────────────────────
# Data Fetch Functions  (mirrors src/tools/api.py signatures)
# ─────────────────────────────────────────────────────────────────

def get_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str = None,  # ignored for MX; kept for API compatibility
) -> list[Price]:
    """
    Fetch daily OHLCV price data for an A-share ticker via MX.

    Parameters
    ----------
    ticker      : A-share ticker, e.g. "300059.SZ"
    start_date : YYYY-MM-DD
    end_date   : YYYY-MM-DD

    Returns
    -------
    list[Price]
    """
    if not _is_ashare(ticker):
        return []  # Let the caller fall back to the original api.py

    code = _ticker_to_code(ticker)
    query = (
        f"{code}近{start_date}至{end_date}每个交易日的开盘价收盘价最高价最低价成交量"
    )
    tables, _, _, err = _mx_query_tables(query)
    if err:
        logger.warning("MX get_prices error for %s: %s", ticker, err)
        return []

    prices: list[Price] = []
    for table in tables:
        fields = table.get("fieldnames", [])
        # Find the sheet that has all required OHLCV fields
        need = {"开盘价", "收盘价", "最高价", "最低价", "成交量"}
        if not need.issubset(set(fields)):
            continue
        for row in table.get("rows", []):
            prices.append(
                Price(
                    open=_parse_chinese_number(row.get("开盘价", "0")),
                    close=_parse_chinese_number(row.get("收盘价", "0")),
                    high=_parse_chinese_number(row.get("最高价", "0")),
                    low=_parse_chinese_number(row.get("最低价", "0")),
                    volume=int(_parse_chinese_number(row.get("成交量", "0"))),
                    time=_clean_date(row.get("date", "")),
                )
            )
    prices.sort(key=lambda p: p.time)
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """
    Fetch financial metrics for an A-share via MX.
    period is ignored (MX returns what's available); limit controls periods returned.
    """
    if not _is_ashare(ticker):
        return []

    code = _ticker_to_code(ticker)
    # MX query: ask for key metrics across multiple periods
    query = (
        f"{code}近{limit}年年度报告的"
        f"净利润、归属于母公司股东的净利润、营业收入、营业总收入、"
        f"资产总计、负债合计、归属于母公司股东权益合计、"
        f"净利润/营业总收入(销售净利率)、销售毛利率、"
        f"净资产收益率ROE、资产负债率、每股收益EPS(基本)"
    )
    tables, _, _, err = _mx_query_tables(query)
    if err:
        logger.warning("MX get_financial_metrics error for %s: %s", ticker, err)
        return []

    metrics: list[FinancialMetrics] = []
    seen_periods: set[str] = set()

    for table in tables:
        fieldnames = table.get("fieldnames", [])
        rows = table.get("rows", [])
        if not rows or "净利润" not in fieldnames:
            continue

        for row in rows:
            period_str = _clean_date(row.get("date", ""))
            if period_str in seen_periods:
                continue
            seen_periods.add(period_str)

            # Use pre-calculated ratios from MX where available
            net_margin_raw = row.get("净利润/营业总收入(销售净利率)", "")
            gross_margin_raw = row.get("销售毛利率", "")
            roe_raw = row.get("净资产收益率ROE", "")
            debt_ratio_raw = row.get("资产负债率", "")
            eps_raw = row.get("每股收益EPS(基本)", "")

            total_assets = _parse_chinese_number(row.get("资产总计", "0"))
            total_debt   = _parse_chinese_number(row.get("负债合计", "0"))
            equity       = total_assets - total_debt
            revenue      = _parse_chinese_number(row.get("营业收入", "0"))

            metrics.append(
                FinancialMetrics(
                    ticker=ticker,
                    report_period=period_str,
                    period=period,
                    currency="CNY",
                    market_cap=None,
                    enterprise_value=None,
                    price_to_earnings_ratio=None,
                    price_to_book_ratio=None,
                    price_to_sales_ratio=None,
                    enterprise_value_to_ebitda_ratio=None,
                    enterprise_value_to_revenue_ratio=None,
                    free_cash_flow_yield=None,
                    peg_ratio=None,
                    gross_margin=_parse_chinese_number(gross_margin_raw),
                    operating_margin=None,
                    net_margin=_parse_chinese_number(net_margin_raw),
                    return_on_equity=_parse_chinese_number(roe_raw),
                    return_on_assets=(net_income / total_assets) if (total_assets and (net_income := _parse_chinese_number(row.get("净利润", "0")))) else None,
                    return_on_invested_capital=None,
                    asset_turnover=None,
                    inventory_turnover=None,
                    receivables_turnover=None,
                    days_sales_outstanding=None,
                    operating_cycle=None,
                    working_capital_turnover=None,
                    current_ratio=None,
                    quick_ratio=None,
                    cash_ratio=None,
                    operating_cash_flow_ratio=None,
                    debt_to_equity=(total_debt / equity) if equity else None,
                    debt_to_assets=_parse_chinese_number(debt_ratio_raw),
                    interest_coverage=None,
                    revenue_growth=None,
                    earnings_growth=None,
                    book_value_growth=None,
                    earnings_per_share_growth=None,
                    free_cash_flow_growth=None,
                    operating_income_growth=None,
                    ebitda_growth=None,
                    payout_ratio=None,
                    earnings_per_share=_parse_chinese_number(eps_raw),
                    book_value_per_share=None,
                    free_cash_flow_per_share=None,
                )
            )

    # Sort by period descending (most recent first) and apply limit
    metrics.sort(key=lambda m: m.report_period, reverse=True)
    return metrics[:limit]


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """
    Fetch financial line items for an A-share via MX.
    line_items is a list of English field names (capital_expenditure, net_income, etc.)
    which are mapped to their Chinese MX equivalents.
    """
    if not _is_ashare(ticker):
        return []

    code = _ticker_to_code(ticker)

    # Map English line item names to Chinese MX queries
    item_map: dict[str, str] = {
        "net_income":                    "净利润",
        "revenue":                       "营业收入",
        "total_assets":                  "资产总计",
        "total_debt":                    "负债合计",
        "shareholders_equity":           "归属于母公司股东权益合计",
        "capital_expenditure":           "购建固定资产无形资产和其他长期资产支付的现金",
        "depreciation_and_amortization": "折旧",
        "gross_profit":                  "毛利润",
        "operating_income":              "营业利润",
        "interest_expense":              "利息支出",
        "cash_and_equivalents":          "货币资金",
        "outstanding_shares":            "发行在外普通股加权平均数",
        "free_cash_flow":                "自由现金流",
        "working_capital":               "营运资本",
        "ebitda":                        "EBITDA",
        "ebit":                          "息税前利润",
    }

    # Build a compact MX query asking for available items
    needed = [item_map.get(k, k) for k in line_items]
    needed_str = "、".join(needed[:8])  # MX has context length limits
    query = f"{code}近{limit}年年度报告的{needed_str}"

    tables, _, _, err = _mx_query_tables(query)
    if err:
        logger.warning("MX search_line_items error for %s: %s", ticker, err)
        return []

    results: list[LineItem] = []
    seen_periods: set[str] = set()

    for table in tables:
        rows = table.get("rows", [])
        if not rows:
            continue
        fieldnames = table.get("fieldnames", [])

        for row in rows:
            period_str = _clean_date(row.get("date", ""))
            if period_str in seen_periods:
                continue
            seen_periods.add(period_str)

            item_data = LineItem(
                ticker=ticker,
                report_period=period_str,
                period=period,
                currency="CNY",
            )
            # Map available fields back
            for eng_name, cn_name in item_map.items():
                if cn_name in fieldnames:
                    setattr(item_data, eng_name, _parse_chinese_number(row.get(cn_name, "0")))
            results.append(item_data)

    results.sort(key=lambda i: i.report_period, reverse=True)
    return results[:limit]


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch latest market cap for an A-share via MX."""
    if not _is_ashare(ticker):
        return None

    code = _ticker_to_code(ticker)
    query = f"{code}最新总市值流通市值"
    tables, _, _, err = _mx_query_tables(query)
    if err:
        logger.warning("MX get_market_cap error for %s: %s", ticker, err)
        return None

    cap: float | None = None
    for table in tables:
        rows = table.get("rows", [])
        if not rows:
            continue
        fieldnames = table.get("fieldnames", [])
        # Prefer 总市值, fall back to 流通市值
        for cap_field in ("总市值", "流通市值", "市值"):
            if cap_field in fieldnames:
                val = _parse_chinese_number(rows[0].get(cap_field, "0"))
                if val > 0:
                    cap = val
                    break
        if cap:
            break
    return cap


def prices_to_df(prices: list[Price]) -> "pd.DataFrame":  # noqa: F821
    """Convert list of Price objects to pandas DataFrame (same as api.py)."""
    import pandas as pd

    df = pd.DataFrame([p.model_dump() for p in prices])
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(
    ticker: str,
    start_date: str,
    end_date: str,
    api_key: str = None,
) -> "pd.DataFrame":  # noqa: F821
    """Get price data as DataFrame (mirrors api.py)."""
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
