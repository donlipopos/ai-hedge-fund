"""
MX Adapter — A-share data layer via 妙想 (MX) API.

Provides the same function signatures as src/tools/api.py so agents can fetch
A-share (沪深) data without any caller-side changes — auto-detects A-share
tickers by .SH/.SZ/.BJ suffix and routes to MX.

Environment:
    MX_APIKEY          — MX API key (also loaded from .env)
    MX_SKILL_PARENT    — parent of mx-data/ skill dir (default: ~/.openclaw/workspace-trading/skills)
"""

from __future__ import annotations

import importlib.util
import os
import re
import json
import logging
from datetime import datetime
from typing import Optional

from src.data.cache import get_cache  # noqa: E402


def _load_mx_data_module():
    """Load mx_data module via file-based import (avoids hyphen directory issue)."""
    # Default to skills/ folder inside the codebase
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_skill_parent = os.path.join(_REPO_ROOT, "skills")
    
    skill_parent = os.path.expandvars(os.path.expanduser(
        os.getenv("MX_SKILL_PARENT", default_skill_parent)
    ))
    module_path = os.path.join(skill_parent, "mx-data", "mx_data.py")
    spec = importlib.util.spec_from_file_location("mx_data", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mx_data_mod = None


def _get_mx_data():
    global _mx_data_mod
    if _mx_data_mod is None:
        _mx_data_mod = _load_mx_data_module()
    return _mx_data_mod


class MXDataWrapper:
    """Thin wrapper to expose MXData.query and MXData.parse_result without importing the class directly."""
    @staticmethod
    def query(query: str) -> dict:
        mod = _get_mx_data()
        return mod.MXData().query(query)

    @staticmethod
    def parse_result(result: dict):
        mod = _get_mx_data()
        return mod.MXData.parse_result(result)

from src.data.models import (
    Price,
    PriceResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    LineItem,
    LineItemResponse,
    CompanyNews,
    InsiderTrade,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _chunk_tickers(tickers: list[str], chunk_size: int = 5) -> list[list[str]]:
    return [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]


def _extract_code_from_string(text: str) -> str | None:
    """Extract a 6-digit stock code from an arbitrary string (e.g. column header or sheet name)."""
    match = re.search(r"(\d{6})", text)
    return match.group(1) if match else None


def _get_mx_client():
    api_key = os.getenv("MX_APIKEY")
    if not api_key:
        raise ValueError("MX_APIKEY environment variable not set.")
    mod = _get_mx_data()
    return mod.MXData(api_key=api_key)


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
    logger.info(f"Querying MX API: {query}")
    result = MXDataWrapper.query(query)
    return MXDataWrapper.parse_result(result)


# ─────────────────────────────────────────────────────────────────
# Cache Warming Functions
# ─────────────────────────────────────────────────────────────────

def warm_market_cap_cache(tickers: list[str], end_date: str) -> None:
    """Pre-fetch and cache market cap for multiple A-share tickers."""
    logger.info(f"Warming market cap cache for {len(tickers)} A-shares...")
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
        
        for table in tables:
            rows = table.get("rows", [])
            for row in rows:
                for col_name, cell_value in row.items():
                    if col_name == "date" or not cell_value:
                        continue

                    code = _extract_code_from_string(col_name)
                    if code and code in code_to_ticker:
                        cap_val = _parse_chinese_number(str(cell_value))
                        if cap_val > 0:  # Ensure we got a valid parse
                            ticker = code_to_ticker[code]
                            cache_key = f"{ticker}_{end_date}"
                            # Only overwrite if it's currently None
                            if cache._market_cap_cache.get(cache_key) is None:
                                cache._market_cap_cache[cache_key] = cap_val

def warm_financial_metrics_cache(tickers: list[str], end_date: str, period: str = "ttm", limit: int = 10) -> None:
    """Pre-fetch and cache financial metrics for multiple A-share tickers."""
    logger.info(f"Warming financial metrics cache for {len(tickers)} A-shares...")
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
            
        query = (
            f"{'和'.join(needed_codes)}近{limit}年年度报告的"
            f"净利润、归属于母公司股东的净利润、营业收入、营业总收入、"
            f"资产总计、负债合计、流动资产合计、流动负债合计、归属于母公司股东权益合计、"
            f"净利润/营业总收入(销售净利率)、销售毛利率、"
            f"净资产收益率ROE、资产负债率、基本每股收益、"
            f"市盈率TTM、市净率、市销率、股息率、营业利润、总股本、"
            f"每股净资产、每股自由现金流、营业收入同比增长、净利润同比增长"
        )
        tables, titles, _, err = _mx_query_tables(query)
        if err or not tables:
            continue
            
        code_to_ticker = {c: t for c, t in zip(needed_codes, needed_tickers)}
        
        # Group rows by ticker
        ticker_metrics: dict[str, list[FinancialMetrics]] = {t: [] for t in needed_tickers}
        
        for i, table in enumerate(tables):
            # Extract code from the title (sheet name)
            title = table.get("sheet_name")
            if not title:
                title = titles[i] if i < len(titles) else ""
            code = _extract_code_from_string(title)
            
            if code and code in code_to_ticker:
                ticker = code_to_ticker[code]
                
                rows = table.get("rows", [])
                fieldnames = table.get("fieldnames", [])
                if not rows or "净利润" not in fieldnames:
                    continue

                seen_periods = set()
                for row in rows:
                    period_str = _clean_date(row.get("date", ""))
                    if period_str in seen_periods:
                        continue
                    seen_periods.add(period_str)

                    # Use pre-calculated ratios from MX where available
                    net_margin_raw = row.get("净利润/营业总收入(销售净利率)") or row.get("销售净利率") or ""
                    gross_margin_raw = row.get("销售毛利率") or ""
                    roe_raw = row.get("净资产收益率ROE(加权)") or row.get("净资产收益率ROE") or row.get("ROE") or row.get("净资产收益率") or ""

                    debt_ratio_raw = row.get("资产负债率") or ""
                    eps_raw = row.get("基本每股收益") or row.get("每股收益") or ""
                    
                    # Map PE, PB, PS, Div Yield
                    pe = _parse_chinese_number(row.get("市盈率TTM") or row.get("PE") or row.get("市盈率(TTM)") or "0")
                    pb = _parse_chinese_number(row.get("市净率") or row.get("PB") or row.get("市净率(PB)") or "0")
                    ps = _parse_chinese_number(row.get("市销率") or row.get("PS") or row.get("市销率(TTM)") or "0")
                    dy = _parse_chinese_number(row.get("股息率") or row.get("股息率(%)") or "0")

                    total_assets = _parse_chinese_number(row.get("资产总计") or row.get("资产总额") or row.get("资产计") or "0")
                    total_liab   = _parse_chinese_number(row.get("负债合计") or row.get("负债总额") or "0")
                    current_liab = _parse_chinese_number(row.get("流动负债合计") or row.get("流动负债") or "0")
                    equity       = _parse_chinese_number(row.get("归属于母公司股东权益合计") or row.get("净资产") or "0") or (total_assets - total_liab)
                    revenue      = _parse_chinese_number(row.get("营业收入") or row.get("营收") or row.get("营业总收入") or "0")
                    op_income    = _parse_chinese_number(row.get("营业利润") or "0")
                    net_income   = _parse_chinese_number(row.get("净利润") or row.get("归母净利润") or "0")
                    shares       = _parse_chinese_number(row.get("总股本") or row.get("发行在外普通股加权平均数") or "0")

                    bvps         = _parse_chinese_number(row.get("每股净资产") or row.get("每股净资产(元)") or "0") or ((equity / shares) if (equity and shares) else None)
                    fcf_ps       = _parse_chinese_number(row.get("每股自由现金流") or row.get("每股自由现金流(元)") or "0")
                    rev_growth   = _parse_chinese_number(row.get("营业收入同比增长") or row.get("营收同比增长") or "0")
                    earn_growth  = _parse_chinese_number(row.get("净利润同比增长") or row.get("归母净利润同比增长") or "0")

                    ticker_metrics[ticker].append(
                        FinancialMetrics(
                            ticker=ticker,
                            report_period=period_str,
                            period=period,
                            currency="CNY",
                            market_cap=None,
                            enterprise_value=None,
                            price_to_earnings_ratio=pe if pe > 0 else None,
                            price_to_book_ratio=pb if pb > 0 else None,
                            price_to_sales_ratio=ps if ps > 0 else None,
                            enterprise_value_to_ebitda_ratio=None,
                            enterprise_value_to_revenue_ratio=None,
                            free_cash_flow_yield=dy if dy > 0 else None,
                            peg_ratio=None,
                            gross_margin=_parse_chinese_number(gross_margin_raw),
                            operating_margin=(op_income / revenue) if revenue else None,
                            net_margin=_parse_chinese_number(net_margin_raw),
                            return_on_equity=_parse_chinese_number(roe_raw),
                            return_on_assets=(net_income / total_assets) if total_assets else None,
                            return_on_invested_capital=(net_income / (total_assets - current_liab)) if (total_assets and total_assets > current_liab) else None,
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
                            debt_to_equity=(total_liab / equity) if equity else None,
                            debt_to_assets=_parse_chinese_number(debt_ratio_raw),
                            interest_coverage=None,
                            revenue_growth=rev_growth,
                            earnings_growth=earn_growth,
                            book_value_growth=None,
                            earnings_per_share_growth=earn_growth,
                            free_cash_flow_growth=None,
                            operating_income_growth=None,
                            ebitda_growth=None,
                            payout_ratio=None,
                            earnings_per_share=_parse_chinese_number(eps_raw) or _parse_chinese_number(row.get("基本每股收益") or row.get("EPS") or "0"),
                            book_value_per_share=bvps,
                            free_cash_flow_per_share=fcf_ps,
                        )
                    )

        # Sort and cache
        for ticker, metrics in ticker_metrics.items():
            if metrics:
                # Sort descending by date
                metrics.sort(key=lambda x: x.report_period, reverse=True)
                result = metrics[:limit]
                cache_key = f"{ticker}_{period}_{end_date}_{limit}"
                cache.set_financial_metrics(cache_key, [m.model_dump() for m in result])


def warm_line_items_cache(
    tickers: list[str],
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> None:
    """Pre-fetch and cache financial line items for multiple A-share tickers."""
    logger.info(f"Warming line items cache for {len(tickers)} A-shares...")
    cache = get_cache()
    ashare_tickers = [t for t in tickers if _is_ashare(t)]
    chunks = _chunk_tickers(ashare_tickers, chunk_size=3)

    # Re-use item_map from search_line_items
    item_map: dict[str, str] = {
        "net_income":                    "净利润 归母净利润",
        "revenue":                       "营业收入 营收",
        "total_assets":                  "资产总计 资产总额",
        "total_liabilities":             "负债合计 负债总额",
        "current_assets":                "流动资产合计 流动资产",
        "current_liabilities":           "流动负债合计 流动负债",
        "shareholders_equity":           "归属于母公司股东权益合计 净资产",
        "capital_expenditure":           "购建固定资产无形资产和其他长期资产支付的现金",
        "depreciation_and_amortization": "折旧",
        "gross_profit":                  "毛利润",
        "operating_income":              "营业利润",
        "interest_expense":              "利息支出",
        "cash_and_equivalents":          "货币资金",
        "outstanding_shares":            "总股本 发行在外普通股加权平均数",
        "free_cash_flow":                "自由现金流",
        "working_capital":               "营运资本",
        "ebitda":                        "EBITDA",
        "ebit":                          "息税前利润",
        "research_and_development":      "研发费用",
        "dividends_and_other_cash_distributions": "分配股利、利润或偿付利息支付的现金",
        "issuance_or_purchase_of_equity_shares": "吸收投资收到的现金 回购股份",
        "earnings_per_share":            "基本每股收益 EPS",
        "book_value_per_share":          "每股净资产",
        "free_cash_flow_per_share":      "每股自由现金流",
        "revenue_growth":                "营业收入同比增长",
        "earnings_growth":               "净利润同比增长",
    }

    for chunk in chunks:
        needed_tickers = []
        needed_codes = []
        for t in chunk:
            cache_key = f"{t}_{period}_{end_date}_{limit}"
            if not cache.get_line_items(cache_key):
                needed_tickers.append(t)
                needed_codes.append(_ticker_to_code(t))

        if not needed_codes:
            continue

        needed_cn = [item_map.get(k, k) for k in line_items]
        # MX context length limit: typically 8-10 items per batch query is safe
        needed_str = "、".join(needed_cn[:10])
        query = f"{'和'.join(needed_codes)}近{limit}年年度报告的{needed_str}"

        tables, titles, _, err = _mx_query_tables(query)
        if err or not tables:
            continue

        code_to_ticker = {c: t for c, t in zip(needed_codes, needed_tickers)}
        ticker_items: dict[str, list[LineItem]] = {t: [] for t in needed_tickers}

        for i, table in enumerate(tables):
            title = table.get("sheet_name")
            if not title:
                title = titles[i] if i < len(titles) else ""
            code = _extract_code_from_string(title)

            if code and code in code_to_ticker:
                ticker = code_to_ticker[code]
                rows = table.get("rows", [])
                if not rows:
                    continue

                seen_periods = set()
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
                    # Map available fields
                    for eng_name, cn_name in item_map.items():
                        if cn_name in row:
                            val = _parse_chinese_number(str(row[cn_name]))
                            setattr(item_data, eng_name, val)
                    
                    # Compute derived fields if not directly returned by MX
                    if item_data.operating_margin is None and item_data.operating_income and item_data.revenue:
                        item_data.operating_margin = item_data.operating_income / item_data.revenue
                    if item_data.roic is None and item_data.net_income and item_data.total_assets and item_data.current_liabilities:
                        item_data.roic = item_data.net_income / (item_data.total_assets - item_data.current_liabilities)
                    if item_data.book_value_per_share is None and item_data.shareholders_equity and item_data.outstanding_shares:
                        item_data.book_value_per_share = item_data.shareholders_equity / item_data.outstanding_shares

                    ticker_items[ticker].append(item_data)

        # Cache results
        for ticker, items in ticker_items.items():
            if items:
                items.sort(key=lambda x: x.report_period, reverse=True)
                result = items[:limit]
                cache_key = f"{ticker}_{period}_{end_date}_{limit}"
                cache.set_line_items(cache_key, [it.model_dump() for it in result])


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

    # Check local cache first (pre-warmed from JSON cache file)
    cache = get_cache()
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached := cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

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
        # Broaden the field matching
        field_map = {
            "open":   next((f for f in fields if f in ("开盘价", "开盘", "Open")), None),
            "close":  next((f for f in fields if f in ("收盘价", "收盘", "Close")), None),
            "high":   next((f for f in fields if f in ("最高价", "最高", "High")), None),
            "low":    next((f for f in fields if f in ("最低价", "最低", "Low")), None),
            "volume": next((f for f in fields if f in ("成交量", "Volume")), None),
            "time":   next((f for f in fields if f in ("date", "日期", "时间", "Time")), None),
        }
        
        if not all([field_map["open"], field_map["close"], field_map["time"]]):
            continue

        for row in table.get("rows", []):
            try:
                prices.append(
                    Price(
                        open=_parse_chinese_number(str(row.get(field_map["open"], "0"))),
                        close=_parse_chinese_number(str(row.get(field_map["close"], "0"))),
                        high=_parse_chinese_number(str(row.get(field_map["high"], "0"))) if field_map["high"] else 0.0,
                        low=_parse_chinese_number(str(row.get(field_map["low"], "0"))) if field_map["low"] else 0.0,
                        volume=int(_parse_chinese_number(str(row.get(field_map["volume"], "0")))) if field_map["volume"] else 0,
                        time=_clean_date(str(row.get(field_map["time"], ""))),
                    )
                )
            except Exception:
                continue
        if prices: 
            break # Found the price table
            
    prices.sort(key=lambda p: p.time)
    if prices:
        cache.set_prices(cache_key, [p.model_dump() for p in prices])
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

    # Check local cache first (pre-warmed from JSON cache file)
    cache = get_cache()
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached := cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached]

    code = _ticker_to_code(ticker)
    # MX query: ask for key metrics across multiple periods
    query = (
        f"{code}近{limit}年年度报告的"
        f"净利润、归属于母公司股东的净利润、营业收入、营业总收入、"
        f"资产总计、负债合计、流动资产合计、流动负债合计、归属于母公司股东权益合计、"
        f"净利润/营业总收入(销售净利率)、销售毛利率、"
        f"净资产收益率ROE、资产负债率、基本每股收益、"
        f"市盈率TTM、市净率、市销率、股息率、营业利润、总股本、"
        f"每股净资产、每股自由现金流、营业收入同比增长、净利润同比增长"
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
            # Skip rows/tables that don't have a valid date/period
            if not period_str or period_str in seen_periods or len(period_str) < 4:
                continue
            seen_periods.add(period_str)

            # Use pre-calculated ratios from MX where available
            net_margin_raw = row.get("净利润/营业总收入(销售净利率)") or row.get("销售净利率") or ""
            gross_margin_raw = row.get("销售毛利率") or ""
            roe_raw = row.get("净资产收益率ROE(加权)") or row.get("净资产收益率ROE") or row.get("ROE") or row.get("净资产收益率") or ""

            debt_ratio_raw = row.get("资产负债率") or ""
            eps_raw = row.get("基本每股收益") or row.get("每股收益") or ""
            
            # Map PE, PB, PS, Div Yield
            pe = _parse_chinese_number(row.get("市盈率TTM") or row.get("PE") or row.get("市盈率(TTM)") or "0")
            pb = _parse_chinese_number(row.get("市净率") or row.get("PB") or row.get("市净率(PB)") or "0")
            ps = _parse_chinese_number(row.get("市销率") or row.get("PS") or row.get("市销率(TTM)") or "0")
            dy = _parse_chinese_number(row.get("股息率") or row.get("股息率(%)") or "0")
            mkt_cap = _parse_chinese_number(row.get("总市值") or row.get("市值") or "0")

            total_assets = _parse_chinese_number(row.get("资产总计") or row.get("资产总额") or row.get("资产计") or "0")
            total_liab   = _parse_chinese_number(row.get("负债合计") or row.get("负债总额") or "0")
            current_liab = _parse_chinese_number(row.get("流动负债合计") or row.get("流动负债") or "0")
            equity       = _parse_chinese_number(row.get("归属于母公司股东权益合计") or row.get("净资产") or "0") or (total_assets - total_liab)
            revenue      = _parse_chinese_number(row.get("营业收入") or row.get("营收") or row.get("营业总收入") or "0")
            op_income    = _parse_chinese_number(row.get("营业利润") or "0")
            net_income   = _parse_chinese_number(row.get("净利润") or row.get("归母净利润") or "0")
            shares       = _parse_chinese_number(row.get("总股本") or row.get("发行在外普通股加权平均数") or "0")

            bvps         = _parse_chinese_number(row.get("每股净资产") or row.get("每股净资产(元)") or "0") or ((equity / shares) if (equity and shares) else None)
            fcf_ps       = _parse_chinese_number(row.get("每股自由现金流") or row.get("每股自由现金流(元)") or "0")
            rev_growth   = _parse_chinese_number(row.get("营业收入同比增长") or row.get("营收同比增长") or "0")
            earn_growth  = _parse_chinese_number(row.get("净利润同比增长") or row.get("归母净利润同比增长") or "0")

            metrics.append(
                FinancialMetrics(
                    ticker=ticker,
                    report_period=period_str,
                    period=period,
                    currency="CNY",
                    market_cap=mkt_cap if mkt_cap > 0 else None,
                    enterprise_value=None,

                    price_to_earnings_ratio=pe if pe > 0 else None,
                    price_to_book_ratio=pb if pb > 0 else None,
                    price_to_sales_ratio=ps if ps > 0 else None,
                    enterprise_value_to_ebitda_ratio=None,
                    enterprise_value_to_revenue_ratio=None,
                    free_cash_flow_yield=dy if dy > 0 else None,
                    peg_ratio=None,
                    gross_margin=_parse_chinese_number(gross_margin_raw),
                    operating_margin=(op_income / revenue) if revenue else None,
                    net_margin=_parse_chinese_number(net_margin_raw),
                    return_on_equity=_parse_chinese_number(roe_raw),
                    return_on_assets=(net_income / total_assets) if total_assets else None,
                    return_on_invested_capital=(net_income / (total_assets - current_liab)) if (total_assets and total_assets > current_liab) else None,
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
                    debt_to_equity=(total_liab / equity) if equity else None,
                    debt_to_assets=_parse_chinese_number(debt_ratio_raw),
                    interest_coverage=None,
                    revenue_growth=rev_growth,
                    earnings_growth=earn_growth,
                    book_value_growth=None,
                    earnings_per_share_growth=earn_growth,
                    free_cash_flow_growth=None,
                    operating_income_growth=None,
                    ebitda_growth=None,
                    payout_ratio=None,
                    earnings_per_share=_parse_chinese_number(eps_raw) or _parse_chinese_number(row.get("基本每股收益") or row.get("EPS") or "0"),
                    book_value_per_share=bvps,
                    free_cash_flow_per_share=fcf_ps,
                )
            )
    # Sort by period descending (most recent first) and apply limit
    metrics.sort(key=lambda m: m.report_period, reverse=True)
    result = metrics[:limit]
    if result:
        cache.set_financial_metrics(cache_key, [m.model_dump() for m in result])
    return result


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
    # Synonyms are joined with space - MX NLP handles this well
    item_map: dict[str, str] = {
        "net_income":                    "净利润 归母净利润",
        "revenue":                       "营业收入 营收",
        "total_assets":                  "资产总计 资产总额",
        "total_liabilities":             "负债合计 负债总额",
        "current_assets":                "流动资产合计 流动资产",
        "current_liabilities":           "流动负债合计 流动负债",
        "shareholders_equity":           "归属于母公司股东权益合计 净资产",
        "capital_expenditure":           "购建固定资产无形资产和其他长期资产支付的现金",
        "depreciation_and_amortization": "折旧",
        "gross_profit":                  "毛利润",
        "operating_income":              "营业利润",
        "interest_expense":              "利息支出",
        "cash_and_equivalents":          "货币资金",
        "outstanding_shares":            "总股本 发行在外普通股加权平均数",
        "free_cash_flow":                "自由现金流",
        "working_capital":               "营运资本",
        "ebitda":                        "EBITDA",
        "ebit":                          "息税前利润",
        "research_and_development":      "研发费用",
        "dividends_and_other_cash_distributions": "分配股利、利润或偿付利息支付的现金",
        "issuance_or_purchase_of_equity_shares": "吸收投资收到的现金 回购股份",
        "earnings_per_share":            "基本每股收益 EPS",
        "book_value_per_share":          "每股净资产",
        "free_cash_flow_per_share":      "每股自由现金流",
        "revenue_growth":                "营业收入同比增长",
        "earnings_growth":               "净利润同比增长",
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
            # Skip rows/tables that don't have a valid date/period
            if not period_str or period_str in seen_periods or len(period_str) < 4:
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
                if cn_name in row:
                    setattr(item_data, eng_name, _parse_chinese_number(str(row[cn_name])))
            
            # Compute derived fields if not directly returned by MX
            if item_data.operating_margin is None and item_data.operating_income and item_data.revenue:
                item_data.operating_margin = item_data.operating_income / item_data.revenue
            if item_data.roic is None and item_data.net_income and item_data.total_assets and item_data.current_liabilities:
                item_data.roic = item_data.net_income / (item_data.total_assets - item_data.current_liabilities)
            if item_data.book_value_per_share is None and item_data.shareholders_equity and item_data.outstanding_shares:
                item_data.book_value_per_share = item_data.shareholders_equity / item_data.outstanding_shares

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

    # Check local cache first (pre-warmed from JSON cache file)
    cache = get_cache()
    cache_key = f"{ticker}_{end_date}"
    # market_cap is stored as a simple float under a dedicated key
    cached_cap = getattr(cache, '_market_cap_cache', {}).get(cache_key)
    if cached_cap is not None:
        return cached_cap

    code = _ticker_to_code(ticker)
    query = f"{code}最新总市值流通市值"
    tables, _, _, err = _mx_query_tables(query)
    if err:
        logger.warning("MX get_market_cap error for %s: %s. Checking metrics fallback...", ticker, err)
        # Fallback: check if we have market cap in the metrics cache
        metrics_key = f"{ticker}_ttm_{end_date}_10"
        if cached_metrics := cache.get_financial_metrics(metrics_key):
            # The FinancialMetrics model might have market_cap as None currently,
            # but let's see if we can find it in the first record
            m = cached_metrics[0]
            if m.get("market_cap"):
                cap = m["market_cap"]
                cache._market_cap_cache[cache_key] = cap
                return cap
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
    if cap is not None:
        if not hasattr(cache, '_market_cap_cache'):
            cache._market_cap_cache = {}
        cache._market_cap_cache[cache_key] = cap
    return cap


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 10,
    api_key: str | None = None,
) -> list[CompanyNews]:
    """Fetch recent news for an A-share ticker via MX."""
    if not _is_ashare(ticker):
        return []
    code = _ticker_to_code(ticker)
    query = f"{code}相关的最新新闻"
    tables, _, _, err = _mx_query_tables(query)
    if err or not tables:
        return []

    results = []
    for table in tables:
        rows = table.get("rows", [])
        if not rows:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = row.get("标题") or row.get("news_title") or ""
            date = row.get("时间") or row.get("publish_date") or ""
            url = row.get("链接") or row.get("url") or ""
            source = row.get("来源") or row.get("source") or ""
            if title:
                try:
                    results.append(CompanyNews(ticker=ticker, title=title, date=date, url=url, source=source))
                except Exception as e:
                    logger.warning(f"Error creating CompanyNews for {ticker}: {e}")
    return results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
) -> list[InsiderTrade]:
    """Fetch insider trading activity for an A-share ticker via MX."""
    if not _is_ashare(ticker):
        return []
    code = _ticker_to_code(ticker)
    query = f"{code}最近的高管持股变动和增减持情况"
    tables, _, _, err = _mx_query_tables(query)
    if err or not tables:
        return []

    results = []
    for table in tables:
        rows = table.get("rows", [])
        if not rows:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            # MX columns vary, use heuristics
            name = row.get("股东名称") or row.get("变动人") or "未知"
            # Map Chinese 'type' to security_title or similar if relevant, 
            # but InsiderTrade doesn't have a generic 'type' field.
            # We will put it in security_title as a hint.
            trade_type = row.get("变动方向") or row.get("交易类型") or ""
            qty = _parse_chinese_number(str(row.get("变动股数") or "0"))
            
            try:
                results.append(InsiderTrade(
                    ticker=ticker,
                    name=name,
                    transaction_shares=qty,
                    security_title=trade_type,
                    filing_date=row.get("公告日期") or row.get("date") or ""
                ))
            except Exception as e:
                logger.warning(f"Error creating InsiderTrade for {ticker}: {e}")
    return results[:limit]


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
