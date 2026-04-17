"""
Liang Wenfeng (梁文锋) Investment Agent

Hybrid founder-operator + quant persona inspired by a China AI platform builder.

Core principles:
1. Innovation should compound into durable platform advantage.
2. Capital efficiency matters; growth without discipline is fragile.
3. Strong balance sheets survive compute and capex cycles.
4. Market structure matters; don't ignore momentum, volatility, or crowding.
5. China-tech resilience earns a premium when supported by execution.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import math
from statistics import mean

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    search_line_items,
)
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


class LiangWenfengSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int
    reasoning: str


def liang_wenfeng_agent(state: AgentState, agent_id: str = "liang_wenfeng_agent"):
    """Analyze tickers through a founder-quant China-tech lens."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data: dict[str, dict] = {}
    final_analysis: dict[str, dict] = {}

    price_start = _price_lookback_start(end_date)

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=8, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "free_cash_flow",
                "capital_expenditure",
                "gross_margin",
                "operating_margin",
                "net_income",
                "research_and_development",
                "total_debt",
                "shareholders_equity",
                "cash_and_equivalents",
                "outstanding_shares",
                "operating_income",
            ],
            end_date,
            period="annual",
            limit=8,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, limit=100, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=20, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching price history")
        prices = get_prices(ticker, price_start, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Scoring innovation moat")
        innovation = analyze_innovation_moat(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Scoring capital efficiency")
        capital = analyze_capital_efficiency(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Scoring founder quality")
        founder = analyze_founder_quality(financial_line_items, insider_trades, company_news)

        progress.update_status(agent_id, ticker, "Scoring quant regime fit")
        quant = analyze_quant_regime(prices)

        progress.update_status(agent_id, ticker, "Scoring China-tech resilience")
        resilience = analyze_china_tech_resilience(ticker, company_news, financial_line_items)

        progress.update_status(agent_id, ticker, "Scoring valuation sanity")
        valuation = analyze_valuation_sanity(metrics, financial_line_items, market_cap)

        total_score = (
            innovation["score"] * 0.25
            + capital["score"] * 0.20
            + founder["score"] * 0.15
            + quant["score"] * 0.20
            + resilience["score"] * 0.10
            + valuation["score"] * 0.10
        )

        if total_score >= 7.0:
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis = {
            "signal": signal,
            "score": round(total_score, 2),
            "max_score": 10,
            "innovation_analysis": innovation,
            "capital_efficiency_analysis": capital,
            "founder_quality_analysis": founder,
            "quant_regime_analysis": quant,
            "china_resilience_analysis": resilience,
            "valuation_analysis": valuation,
        }
        analysis_data[ticker] = analysis

        confidence_hint = compute_confidence(analysis)
        progress.update_status(agent_id, ticker, "Generating Liang Wenfeng analysis")
        liang_output = generate_liang_output(
            ticker=ticker,
            analysis_data=analysis,
            state=state,
            agent_id=agent_id,
            confidence_hint=confidence_hint,
        )

        final_analysis[ticker] = {
            "signal": liang_output.signal,
            "confidence": liang_output.confidence,
            "reasoning": liang_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=liang_output.reasoning)

    message = HumanMessage(content=json.dumps(final_analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(final_analysis, "Liang Wenfeng Agent (梁文锋)")

    state["data"]["analyst_signals"][agent_id] = final_analysis
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}


def _price_lookback_start(end_date: str) -> str:
    try:
        return (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    except ValueError:
        return end_date


def _get(obj, field: str, default=None):
    return getattr(obj, field, default)


def _latest_non_null(values: list[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _series(line_items: list, field: str) -> list[float]:
    values: list[float] = []
    for item in line_items:
        value = _get(item, field)
        if value is not None:
            values.append(value)
    return values


def _bounded(score: float) -> float:
    return round(max(0.0, min(10.0, score)), 2)


def analyze_innovation_moat(metrics: list, financial_line_items: list) -> dict:
    score = 0.0
    details = []

    revenues = _series(financial_line_items, "revenue")
    rd_values = _series(financial_line_items, "research_and_development")
    gross_margins = [m for m in _series(financial_line_items, "gross_margin") if m is not None]
    metric_growth = [m.revenue_growth for m in metrics if _get(m, "revenue_growth") is not None]

    if revenues and rd_values and revenues[0] > 0:
        rd_ratio = rd_values[0] / revenues[0]
        if rd_ratio >= 0.12:
            score += 3
            details.append(f"High R&D intensity at {rd_ratio:.1%}")
        elif rd_ratio >= 0.06:
            score += 2
            details.append(f"Healthy R&D intensity at {rd_ratio:.1%}")
        elif rd_ratio > 0:
            score += 1
            details.append(f"Some R&D reinvestment at {rd_ratio:.1%}")
    else:
        details.append("Limited R&D visibility")

    if gross_margins:
        avg_margin = mean(gross_margins)
        margin_range = max(gross_margins) - min(gross_margins)
        if avg_margin >= 0.50 and margin_range <= 0.10:
            score += 3
            details.append(f"Strong margin moat with {avg_margin:.1%} average gross margin")
        elif avg_margin >= 0.35:
            score += 2
            details.append(f"Solid gross margin profile at {avg_margin:.1%}")
        elif avg_margin >= 0.20:
            score += 1
            details.append(f"Moderate gross margin profile at {avg_margin:.1%}")
    else:
        details.append("Gross margin history is sparse")

    growth = _latest_non_null(metric_growth)
    if growth is not None:
        if growth >= 0.20:
            score += 2
            details.append(f"Revenue still compounding at {growth:.1%}")
        elif growth >= 0.08:
            score += 1
            details.append(f"Revenue growth remains positive at {growth:.1%}")
    elif len(revenues) >= 2 and revenues[-1] > 0:
        revenue_change = (revenues[0] / revenues[-1]) - 1
        if revenue_change >= 0.40:
            score += 2
            details.append(f"Historical revenue scaled by {revenue_change:.0%}")
        elif revenue_change >= 0.10:
            score += 1
            details.append(f"Historical revenue grew by {revenue_change:.0%}")
    else:
        details.append("Growth history is thin")

    return {"score": _bounded(score), "details": "; ".join(details)}


def analyze_capital_efficiency(metrics: list, financial_line_items: list) -> dict:
    score = 0.0
    details = []

    fcf_values = _series(financial_line_items, "free_cash_flow")
    revenues = _series(financial_line_items, "revenue")
    capex_values = [abs(v) for v in _series(financial_line_items, "capital_expenditure")]
    share_counts = _series(financial_line_items, "outstanding_shares")

    roic = _latest_non_null([_get(m, "return_on_invested_capital") for m in metrics])
    roe = _latest_non_null([_get(m, "return_on_equity") for m in metrics])

    if fcf_values:
        positive_ratio = sum(1 for value in fcf_values if value > 0) / len(fcf_values)
        if positive_ratio == 1:
            score += 3
            details.append("Free cash flow is positive in every observed period")
        elif positive_ratio >= 0.6:
            score += 2
            details.append(f"FCF positive in {positive_ratio:.0%} of periods")
        else:
            details.append("Free cash flow generation is inconsistent")
    else:
        details.append("No free cash flow history")

    if capex_values and revenues and revenues[0] > 0:
        capex_ratio = capex_values[0] / revenues[0]
        if capex_ratio <= 0.06:
            score += 2
            details.append(f"Capex discipline at {capex_ratio:.1%} of revenue")
        elif capex_ratio <= 0.12:
            score += 1
            details.append(f"Moderate capex intensity at {capex_ratio:.1%}")
        else:
            details.append(f"Heavy capex load at {capex_ratio:.1%}")
    else:
        details.append("Capex intensity unavailable")

    if roic is not None or roe is not None:
        quality_return = roic if roic is not None else roe
        if quality_return is not None and quality_return >= 0.15:
            score += 3
            details.append(f"High return on capital at {quality_return:.1%}")
        elif quality_return is not None and quality_return >= 0.08:
            score += 2
            details.append(f"Reasonable return on capital at {quality_return:.1%}")
        elif quality_return is not None and quality_return > 0:
            score += 1
            details.append(f"Low but positive return on capital at {quality_return:.1%}")
    else:
        details.append("Return on capital not available")

    if len(share_counts) >= 2 and share_counts[-1] > 0:
        dilution = (share_counts[0] / share_counts[-1]) - 1
        if dilution <= 0:
            score += 2
            details.append("No dilution pressure; share count is flat or lower")
        elif dilution <= 0.08:
            score += 1
            details.append(f"Manageable dilution at {dilution:.0%}")
        else:
            details.append(f"Meaningful dilution at {dilution:.0%}")
    else:
        details.append("Share-count discipline unclear")

    return {"score": _bounded(score), "details": "; ".join(details)}


def analyze_founder_quality(financial_line_items: list, insider_trades: list, company_news: list) -> dict:
    score = 0.0
    details = []

    debt_values = _series(financial_line_items, "total_debt")
    equity_values = _series(financial_line_items, "shareholders_equity")
    fcf_values = _series(financial_line_items, "free_cash_flow")
    ni_values = _series(financial_line_items, "net_income")

    if debt_values and equity_values and equity_values[0] > 0:
        debt_to_equity = debt_values[0] / equity_values[0]
        if debt_to_equity <= 0.2:
            score += 3
            details.append(f"Low leverage with D/E of {debt_to_equity:.2f}")
        elif debt_to_equity <= 0.6:
            score += 2
            details.append(f"Contained leverage with D/E of {debt_to_equity:.2f}")
        elif debt_to_equity <= 1.0:
            score += 1
            details.append(f"Leverage is acceptable but not clean at D/E {debt_to_equity:.2f}")
        else:
            details.append(f"Leverage is high at D/E {debt_to_equity:.2f}")
    else:
        details.append("Balance-sheet discipline unclear")

    ratios = [
        fcf_values[i] / ni_values[i]
        for i in range(min(len(fcf_values), len(ni_values)))
        if ni_values[i] and ni_values[i] > 0
    ]
    if ratios:
        conversion = mean(ratios)
        if conversion >= 1.0:
            score += 3
            details.append(f"Cash conversion is strong at FCF/NI {conversion:.2f}")
        elif conversion >= 0.7:
            score += 2
            details.append(f"Cash conversion is acceptable at FCF/NI {conversion:.2f}")
        elif conversion > 0:
            score += 1
            details.append(f"Cash conversion is weak at FCF/NI {conversion:.2f}")
    else:
        details.append("Cash conversion cannot be assessed")

    buy_count = 0
    sell_count = 0
    for trade in insider_trades:
        txn_type = str(_get(trade, "transaction_type", "") or "").lower()
        if "buy" in txn_type or "purchase" in txn_type:
            buy_count += 1
        elif "sell" in txn_type or "sale" in txn_type:
            sell_count += 1
    if buy_count or sell_count:
        buy_ratio = buy_count / (buy_count + sell_count)
        if buy_ratio >= 0.6:
            score += 2
            details.append(f"Insiders are net buyers in {buy_ratio:.0%} of observed trades")
        elif buy_ratio >= 0.4:
            score += 1
            details.append(f"Insider activity is balanced with {buy_ratio:.0%} buys")
        else:
            details.append("Insider flow leans toward selling")
    else:
        details.append("No usable insider signal")

    positive_news_hits = 0
    caution_news_hits = 0
    for item in company_news:
        title = f"{_get(item, 'title', '')} {_get(item, 'source', '')}".lower()
        if any(keyword in title for keyword in ["open source", "developer", "launch", "breakthrough", "model", "platform", "efficiency"]):
            positive_news_hits += 1
        if any(keyword in title for keyword in ["probe", "fraud", "lawsuit", "ban", "sanction", "delay"]):
            caution_news_hits += 1
    if positive_news_hits > caution_news_hits and positive_news_hits >= 2:
        score += 2
        details.append("News flow points to execution and product momentum")
    elif caution_news_hits > positive_news_hits and caution_news_hits >= 2:
        details.append("News flow raises execution or governance questions")

    return {"score": _bounded(score), "details": "; ".join(details)}


def analyze_quant_regime(prices: list) -> dict:
    score = 0.0
    details = []

    closes = [float(_get(price, "close")) for price in prices if _get(price, "close") is not None]
    if len(closes) < 20:
        return {"score": 3.0, "details": "Insufficient price history; quant regime defaults to cautious neutral"}

    latest = closes[-1]
    momentum_1m = (latest / closes[-21]) - 1 if len(closes) >= 21 and closes[-21] else 0.0
    momentum_3m = (latest / closes[-64]) - 1 if len(closes) >= 64 and closes[-64] else momentum_1m
    momentum_6m = (latest / closes[-127]) - 1 if len(closes) >= 127 and closes[-127] else momentum_3m

    returns = []
    for prev, current in zip(closes[:-1], closes[1:]):
        if prev:
            returns.append((current / prev) - 1)

    realized_vol = math.sqrt(252) * _stdev(returns) if returns else 0.0
    peak = closes[0]
    max_drawdown = 0.0
    for close in closes:
        peak = max(peak, close)
        if peak:
            max_drawdown = min(max_drawdown, (close / peak) - 1)

    if momentum_3m > 0.15 and momentum_6m > 0.20:
        score += 4
        details.append(f"Momentum is strong across 3-6 months ({momentum_3m:.1%}/{momentum_6m:.1%})")
    elif momentum_3m > 0.05:
        score += 2.5
        details.append(f"Momentum is constructive at {momentum_3m:.1%} over 3 months")
    elif momentum_3m > -0.05:
        score += 1.5
        details.append("Momentum is range-bound")
    else:
        details.append(f"Momentum is weak at {momentum_3m:.1%} over 3 months")

    if realized_vol <= 0.35:
        score += 3
        details.append(f"Volatility is controlled at {realized_vol:.1%}")
    elif realized_vol <= 0.55:
        score += 1.5
        details.append(f"Volatility is elevated but manageable at {realized_vol:.1%}")
    else:
        details.append(f"Volatility is too high at {realized_vol:.1%}")

    if max_drawdown >= -0.20:
        score += 3
        details.append(f"Drawdown containment is solid at {max_drawdown:.1%}")
    elif max_drawdown >= -0.35:
        score += 1.5
        details.append(f"Drawdown is acceptable at {max_drawdown:.1%}")
    else:
        details.append(f"Deep drawdown at {max_drawdown:.1%}")

    if momentum_1m < -0.10 and momentum_3m > 0:
        score -= 1
        details.append("Short-term reversal risk is building")

    return {"score": _bounded(score), "details": "; ".join(details)}


def analyze_china_tech_resilience(ticker: str, company_news: list, financial_line_items: list) -> dict:
    score = 0.0
    details = []

    if ticker.upper().endswith((".SH", ".SZ", ".BJ", ".HK")):
        score += 3
        details.append("Listed in a China-linked market")

    keyword_hits = 0
    caution_hits = 0
    for item in company_news:
        text = f"{_get(item, 'title', '')} {_get(item, 'source', '')}".lower()
        if any(keyword in text for keyword in ["china", "chinese", "domestic", "国产", "自主", "supply chain", "localization", "semiconductor", "ai"]):
            keyword_hits += 1
        if any(keyword in text for keyword in ["sanction", "restriction", "export control", "delisting"]):
            caution_hits += 1
    if keyword_hits >= 3:
        score += 4
        details.append("News flow points to China-tech positioning or localization tailwinds")
    elif keyword_hits >= 1:
        score += 2
        details.append("Some evidence of China-tech relevance")
    else:
        details.append("China-tech positioning is not explicit in available news")

    rd_values = _series(financial_line_items, "research_and_development")
    if rd_values and rd_values[0] > 0:
        score += 2
        details.append("Internal R&D spending supports resilience")

    if caution_hits >= 2:
        score -= 2
        details.append("Headline risk from restrictions or sanctions is non-trivial")

    return {"score": _bounded(score), "details": "; ".join(details)}


def analyze_valuation_sanity(metrics: list, financial_line_items: list, market_cap: float | None) -> dict:
    score = 0.0
    details = []

    latest = metrics[0] if metrics else None
    ps_ratio = _get(latest, "price_to_sales_ratio")
    peg_ratio = _get(latest, "peg_ratio")
    fcf_values = _series(financial_line_items, "free_cash_flow")

    if ps_ratio is not None:
        if ps_ratio <= 4:
            score += 3
            details.append(f"Price-to-sales is reasonable at {ps_ratio:.1f}x")
        elif ps_ratio <= 8:
            score += 2
            details.append(f"Price-to-sales is acceptable at {ps_ratio:.1f}x")
        elif ps_ratio <= 12:
            score += 1
            details.append(f"Price-to-sales is rich at {ps_ratio:.1f}x")
        else:
            details.append(f"Price-to-sales is stretched at {ps_ratio:.1f}x")
    else:
        details.append("No price-to-sales multiple available")

    if peg_ratio is not None:
        if peg_ratio <= 1.5:
            score += 3
            details.append(f"PEG is healthy at {peg_ratio:.2f}")
        elif peg_ratio <= 2.5:
            score += 2
            details.append(f"PEG is workable at {peg_ratio:.2f}")
        elif peg_ratio <= 4.0:
            score += 1
            details.append(f"PEG is demanding at {peg_ratio:.2f}")
        else:
            details.append(f"PEG is expensive at {peg_ratio:.2f}")
    else:
        details.append("PEG is unavailable")

    if market_cap and market_cap > 0 and fcf_values:
        normalized_fcf = mean(fcf_values[: min(3, len(fcf_values))])
        if normalized_fcf > 0:
            fcf_yield = normalized_fcf / market_cap
            if fcf_yield >= 0.05:
                score += 4
                details.append(f"FCF yield is attractive at {fcf_yield:.1%}")
            elif fcf_yield >= 0.025:
                score += 3
                details.append(f"FCF yield is acceptable at {fcf_yield:.1%}")
            elif fcf_yield >= 0.01:
                score += 1
                details.append(f"FCF yield is thin at {fcf_yield:.1%}")
            else:
                details.append(f"FCF yield is very thin at {fcf_yield:.1%}")
        else:
            details.append("FCF is not positive enough to support valuation")
    else:
        details.append("Market-cap or FCF data is incomplete")

    return {"score": _bounded(score), "details": "; ".join(details)}


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def compute_confidence(analysis: dict) -> int:
    score = float(analysis.get("score") or 0.0)
    signal = analysis.get("signal") or "neutral"

    base = 50 + int(abs(score - 5.5) * 9)
    if signal == "bearish":
        base -= 3

    components = [
        analysis.get("innovation_analysis"),
        analysis.get("capital_efficiency_analysis"),
        analysis.get("founder_quality_analysis"),
        analysis.get("quant_regime_analysis"),
        analysis.get("china_resilience_analysis"),
        analysis.get("valuation_analysis"),
    ]
    populated = sum(1 for component in components if component and component.get("details"))
    completeness_penalty = (len(components) - populated) * 5

    if "insufficient" in (analysis.get("quant_regime_analysis", {}).get("details", "").lower()):
        completeness_penalty += 8

    confidence = max(20, min(95, base - completeness_penalty))
    return confidence


def make_liang_facts_bundle(analysis: dict) -> dict:
    innovation = analysis.get("innovation_analysis") or {}
    capital = analysis.get("capital_efficiency_analysis") or {}
    founder = analysis.get("founder_quality_analysis") or {}
    quant = analysis.get("quant_regime_analysis") or {}
    resilience = analysis.get("china_resilience_analysis") or {}
    valuation = analysis.get("valuation_analysis") or {}

    return {
        "pre_signal": analysis.get("signal"),
        "total_score": round(float(analysis.get("score") or 0), 2),
        "innovation_score": round(float(innovation.get("score") or 0), 2),
        "capital_efficiency_score": round(float(capital.get("score") or 0), 2),
        "founder_quality_score": round(float(founder.get("score") or 0), 2),
        "quant_regime_score": round(float(quant.get("score") or 0), 2),
        "china_resilience_score": round(float(resilience.get("score") or 0), 2),
        "valuation_score": round(float(valuation.get("score") or 0), 2),
        "notes": {
            "innovation": (innovation.get("details") or "")[:180],
            "capital_efficiency": (capital.get("details") or "")[:180],
            "founder_quality": (founder.get("details") or "")[:180],
            "quant_regime": (quant.get("details") or "")[:180],
            "china_resilience": (resilience.get("details") or "")[:180],
            "valuation": (valuation.get("details") or "")[:180],
        },
    }


def generate_liang_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    confidence_hint: int,
) -> LiangWenfengSignal:
    facts_bundle = make_liang_facts_bundle(analysis_data)

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are Liang Wenfeng (梁文锋), a disciplined China AI founder-quant persona. "
                "Think in systems: model capability, platform leverage, capital efficiency, and regime awareness. "
                "Do not hype. Prefer blunt, high-signal language. Use only the provided facts. "
                "Return JSON only. Keep reasoning under 160 characters. Use the provided confidence exactly.",
            ),
            (
                "human",
                "Ticker: {ticker}\n"
                "Facts:\n{facts}\n"
                "Confidence: {confidence}\n"
                "Return exactly:\n"
                "{{\n"
                '  "signal": "bullish" | "bearish" | "neutral",\n'
                f'  "confidence": {confidence_hint},\n'
                '  "reasoning": "short justification in Liang Wenfeng\'s voice"\n'
                "}}",
            ),
        ]
    )

    prompt = template.invoke(
        {
            "ticker": ticker,
            "facts": json.dumps(facts_bundle, separators=(",", ":"), ensure_ascii=False),
            "confidence": confidence_hint,
        }
    )

    def _default():
        return LiangWenfengSignal(
            signal=analysis_data.get("signal", "neutral"),
            confidence=confidence_hint,
            reasoning=build_fallback_reasoning(analysis_data),
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=LiangWenfengSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )


def build_fallback_reasoning(analysis: dict) -> str:
    signal = analysis.get("signal", "neutral")
    score = float(analysis.get("score") or 0.0)

    if signal == "bullish":
        lead = "Innovation quality and market structure both support upside"
    elif signal == "bearish":
        lead = "The setup is too fragile on quality, valuation, or regime"
    else:
        lead = "Signals are mixed, so the edge is not clean enough"

    return f"{lead}; composite score {score:.1f}/10."
