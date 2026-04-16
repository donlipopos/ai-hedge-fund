"""
Duan Yongping (段永平) Investment Agent

Investing philosophy distilled from Duan Yongping (雪球: 大道无形我有型):
- Founder of Bubu Hi-Tech (步步高), spiritual father of OPPO and vivo
- Early/major investor in Apple, Tencent, Pinduoduo
- Deep Buffett/Munger disciple with Chinese entrepreneur operational lens

Core principles:
1. "做对的事，把事情做对" - Right direction × right execution
2. "不为清单" - The Stop Doing List (what NOT to do matters more)
3. Owner earnings (FCF) over accounting profits
4. "本分" - Moral discipline within one's role; first filter for management
5. Business-operator's eye: would I want to run this company?
6. Zero leverage, ever.
7. Only invest in what you deeply understand.
8. Time horizon: 10 years minimum.
"""

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items, get_insider_trades, get_company_news
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.utils.api_key import get_api_key_from_state


class DuanYongpingSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int
    reasoning: str


def duan_yongping_agent(state: AgentState, agent_id: str = "duan_yongping_agent"):
    """
    Analyzes stocks using Duan Yongping's investing principles.

    Duan Yongping is a Chinese entrepreneur-investor who founded Bubu Hi-Tech (步步高)
    and is the spiritual father of OPPO and vivo. He moved to Palo Alto in 2001 and
    became a highly successful long-term value investor (major Apple, Tencent, PDD positions).

    His framework differs from pure Buffett/Munger in that he applies an OPERATOR's eye —
    he asks not just "is this a good business?" but "would I want to run this business?"

    Key scoring dimensions:
    1. Business quality & moat ("本分" test — does the business genuinely serve users?)
    2. "不为清单" red flags (leveraged, sacrifices margin for share, poor capital discipline)
    3. FCF quality (owner earnings, not accounting profits)
    4. Management character (candor, integrity, long-term orientation)
    5. 10-year predictability
    6. Valuation reasonableness
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analysis_data = {}
    duan_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching financial metrics")
        # Duan looks at long periods — business quality reveals itself over time
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "net_income",
                "operating_income",
                "free_cash_flow",
                "capital_expenditure",
                "total_debt",
                "cash_and_equivalents",
                "shareholders_equity",
                "gross_margin",
                "operating_margin",
                "return_on_invested_capital",
                "outstanding_shares",
                "research_and_development",
            ],
            end_date,
            period="annual",
            limit=10,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date, limit=100, api_key=api_key)

        progress.update_status(agent_id, ticker, "Fetching company news")
        company_news = get_company_news(ticker, end_date, limit=10, api_key=api_key)

        progress.update_status(agent_id, ticker, "Running 本分 business quality test")
        benfan_analysis = analyze_benfan_quality(metrics, financial_line_items)

        progress.update_status(agent_id, ticker, "Scanning 不为清单 red flags")
        stop_doing_analysis = scan_stop_doing_list(financial_line_items, metrics)

        progress.update_status(agent_id, ticker, "Analyzing FCF / owner earnings")
        fcf_analysis = analyze_owner_earnings(financial_line_items, market_cap)

        progress.update_status(agent_id, ticker, "Evaluating management character")
        management_analysis = evaluate_management_character(financial_line_items, insider_trades)

        progress.update_status(agent_id, ticker, "Assessing 10-year predictability")
        predictability_analysis = assess_ten_year_predictability(financial_line_items)

        # Duan's weighting: business quality and NOT doing wrong things matter most
        # He would NEVER invest in a company with serious 不为 violations, regardless of price
        stop_doing_penalty = stop_doing_analysis["penalty"]

        total_score = (
            benfan_analysis["score"] * 0.30 +      # Core business quality
            fcf_analysis["score"] * 0.30 +          # Owner earnings (his primary metric)
            management_analysis["score"] * 0.20 +   # Management character
            predictability_analysis["score"] * 0.20 # 10-year visibility
        ) * (1 - stop_doing_penalty)                # Penalize hard for 不为 violations

        # Duan has high standards for "wonderful businesses"
        if total_score >= 7.0:
            signal = "bullish"
        elif total_score <= 4.5:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": 10,
            "benfan_analysis": benfan_analysis,
            "stop_doing_analysis": stop_doing_analysis,
            "fcf_analysis": fcf_analysis,
            "management_analysis": management_analysis,
            "predictability_analysis": predictability_analysis,
        }

        progress.update_status(agent_id, ticker, "Generating Duan Yongping analysis")
        duan_output = generate_duan_output(
            ticker=ticker,
            analysis_data=analysis_data[ticker],
            state=state,
            agent_id=agent_id,
            confidence_hint=compute_confidence(analysis_data[ticker], signal),
        )

        duan_analysis[ticker] = {
            "signal": duan_output.signal,
            "confidence": duan_output.confidence,
            "reasoning": duan_output.reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=duan_output.reasoning)

    message = HumanMessage(
        content=json.dumps(duan_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(duan_analysis, "Duan Yongping Agent (段永平)")

    progress.update_status(agent_id, None, "Done")
    state["data"]["analyst_signals"][agent_id] = duan_analysis

    return {
        "messages": [message],
        "data": state["data"],
    }


# ---------------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# ---------------------------------------------------------------------------

def analyze_benfan_quality(metrics: list, financial_line_items: list) -> dict:
    """
    '本分' quality test — does the business genuinely serve users and earn returns?

    Duan asks: "Is this business doing the RIGHT thing?"
    Proxies: high/stable ROIC (value creation), pricing power (gross margin), low capex needs.

    A business that genuinely serves users tends to earn high returns on capital
    consistently — this is the mathematical proof of 本分.
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": 0, "details": "Insufficient data for 本分 quality analysis"}

    # 1. ROIC — the ultimate test of genuine value creation
    roic_values = [
        item.return_on_invested_capital
        for item in financial_line_items
        if hasattr(item, "return_on_invested_capital") and item.return_on_invested_capital is not None
    ]
    if roic_values:
        consistently_high = sum(1 for r in roic_values if r > 0.15)
        ratio = consistently_high / len(roic_values)
        if ratio >= 0.8:
            score += 3
            details.append(f"Excellent ROIC >15% in {consistently_high}/{len(roic_values)} years — genuine value creator")
        elif ratio >= 0.5:
            score += 2
            details.append(f"Good ROIC >15% in {consistently_high}/{len(roic_values)} years")
        elif ratio > 0:
            score += 1
            details.append(f"Inconsistent ROIC >15% in only {consistently_high}/{len(roic_values)} years")
        else:
            details.append("ROIC never exceeds 15% — questions whether this business truly creates value")
    else:
        details.append("No ROIC data available")

    # 2. Gross margin — pricing power = users genuinely value the product
    gross_margins = [
        item.gross_margin
        for item in financial_line_items
        if hasattr(item, "gross_margin") and item.gross_margin is not None
    ]
    if gross_margins:
        avg_gm = sum(gross_margins) / len(gross_margins)
        improving = sum(1 for i in range(1, len(gross_margins)) if gross_margins[i] >= gross_margins[i-1])
        if avg_gm > 0.40 and improving >= len(gross_margins) * 0.6:
            score += 3
            details.append(f"High pricing power: avg gross margin {avg_gm:.1%}, improving trend")
        elif avg_gm > 0.30:
            score += 2
            details.append(f"Solid gross margin {avg_gm:.1%}")
        elif avg_gm > 0.20:
            score += 1
            details.append(f"Moderate gross margin {avg_gm:.1%}")
        else:
            details.append(f"Low gross margin {avg_gm:.1%} — suggests commodity business or weak moat")
    else:
        details.append("No gross margin data")

    # 3. Capex intensity — Duan loves asset-light businesses (like Apple's ecosystem)
    capex_ratios = []
    for item in financial_line_items:
        if (hasattr(item, "capital_expenditure") and item.capital_expenditure is not None and
                hasattr(item, "revenue") and item.revenue and item.revenue > 0):
            capex_ratios.append(abs(item.capital_expenditure) / item.revenue)

    if capex_ratios:
        avg_capex = sum(capex_ratios) / len(capex_ratios)
        if avg_capex < 0.04:
            score += 2
            details.append(f"Asset-light model: avg capex only {avg_capex:.1%} of revenue")
        elif avg_capex < 0.08:
            score += 1
            details.append(f"Moderate capital requirements: {avg_capex:.1%} of revenue")
        else:
            details.append(f"Capital-intensive: {avg_capex:.1%} of revenue — less attractive to Duan")
    else:
        details.append("No capex data")

    # 4. R&D investment — genuine product/service improvement
    rd_values = [
        item.research_and_development
        for item in financial_line_items
        if hasattr(item, "research_and_development") and item.research_and_development is not None
    ]
    if rd_values and sum(rd_values) > 0:
        score += 1
        details.append("Invests in R&D — signal of product focus")

    final_score = min(10, score * 10 / 9)
    return {"score": final_score, "details": "; ".join(details)}


def scan_stop_doing_list(financial_line_items: list, metrics: list) -> dict:
    """
    Duan's '不为清单' — scan for things he would NEVER invest in.

    Violations result in a multiplicative penalty on the total score.
    Duan says: "Knowing what NOT to do matters more than knowing what to do."

    Red flags:
    - High leverage (interest-bearing debt >> equity)
    - Sacrificing margin for market share (revenue growing, margins collapsing)
    - Negative FCF despite reported profits (accounting games)
    - Massive share dilution (management enriching themselves)
    """
    violations = []
    penalty = 0.0

    if not financial_line_items:
        return {"violations": ["No data to scan"], "penalty": 0.0, "details": "Insufficient data"}

    # Red flag 1: High leverage — Duan built Bubu with ZERO interest-bearing debt
    debt_values = [
        item.total_debt
        for item in financial_line_items
        if hasattr(item, "total_debt") and item.total_debt is not None
    ]
    equity_values = [
        item.shareholders_equity
        for item in financial_line_items
        if hasattr(item, "shareholders_equity") and item.shareholders_equity is not None
    ]
    if debt_values and equity_values:
        de_ratio = debt_values[0] / equity_values[0] if equity_values[0] > 0 else float("inf")
        if de_ratio > 2.0:
            violations.append(f"Dangerous leverage: D/E ratio {de_ratio:.1f}x — Duan would not touch this")
            penalty += 0.30
        elif de_ratio > 1.0:
            violations.append(f"High leverage: D/E ratio {de_ratio:.1f}x — caution")
            penalty += 0.10

    # Red flag 2: Sacrificing margin for share — anti-本分 behavior
    revenues = [
        item.revenue for item in financial_line_items
        if hasattr(item, "revenue") and item.revenue is not None
    ]
    op_margins = [
        item.operating_margin for item in financial_line_items
        if hasattr(item, "operating_margin") and item.operating_margin is not None
    ]
    if revenues and op_margins and len(revenues) >= 4 and len(op_margins) >= 4:
        rev_growth = (revenues[0] / revenues[-1]) - 1 if revenues[-1] > 0 else 0
        margin_trend = op_margins[0] - op_margins[-1]  # negative = margin declined
        if rev_growth > 0.5 and margin_trend < -0.10:
            violations.append(f"Revenue growing ({rev_growth:.0%}) but margins collapsing ({margin_trend:.1%}) — burning cash for share")
            penalty += 0.20

    # Red flag 3: Persistent negative FCF despite profits — accounting concern
    fcf_values = [
        item.free_cash_flow for item in financial_line_items
        if hasattr(item, "free_cash_flow") and item.free_cash_flow is not None
    ]
    net_income_values = [
        item.net_income for item in financial_line_items
        if hasattr(item, "net_income") and item.net_income is not None
    ]
    if fcf_values and net_income_values:
        negative_fcf_count = sum(1 for f in fcf_values if f < 0)
        if negative_fcf_count >= len(fcf_values) * 0.7:
            violations.append(f"FCF negative in {negative_fcf_count}/{len(fcf_values)} years despite reported profits — questions real earnings")
            penalty += 0.20

    # Red flag 4: Massive share dilution — management self-enrichment
    shares = [
        item.outstanding_shares for item in financial_line_items
        if hasattr(item, "outstanding_shares") and item.outstanding_shares is not None
    ]
    if shares and len(shares) >= 3:
        dilution = (shares[0] / shares[-1]) - 1
        if dilution > 0.30:  # > 30% more shares outstanding
            violations.append(f"Severe dilution: shares grew {dilution:.0%} — management not aligned with shareholders")
            penalty += 0.15

    # Cap penalty at 60%
    penalty = min(0.60, penalty)

    details = "; ".join(violations) if violations else "No 不为清单 red flags detected — passes basic discipline test"
    return {"violations": violations, "penalty": penalty, "details": details}


def analyze_owner_earnings(financial_line_items: list, market_cap: float) -> dict:
    """
    FCF / Owner Earnings analysis — Duan's PRIMARY valuation metric.

    Unlike P/E based investors, Duan focuses purely on free cash flow.
    "A business that earns money but generates no cash is not my kind of business."

    Valuation: Duan thinks in terms of FCF yield and long-term earnings trajectory,
    not precise DCF models. He wants to roughly know the answer without modeling.
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": 0, "details": "No financial data"}

    fcf_values = [
        item.free_cash_flow for item in financial_line_items
        if hasattr(item, "free_cash_flow") and item.free_cash_flow is not None
    ]

    if not fcf_values or len(fcf_values) < 3:
        return {"score": 0, "details": "Insufficient FCF history (need 3+ years)"}

    # 1. FCF consistency — does this business reliably turn revenue into cash?
    positive_count = sum(1 for f in fcf_values if f > 0)
    if positive_count == len(fcf_values):
        score += 3
        details.append("FCF positive every single year — highly reliable cash generator")
    elif positive_count >= len(fcf_values) * 0.8:
        score += 2
        details.append(f"FCF positive in {positive_count}/{len(fcf_values)} years — mostly reliable")
    elif positive_count >= len(fcf_values) * 0.5:
        score += 1
        details.append(f"FCF mixed: positive in only {positive_count}/{len(fcf_values)} years")
    else:
        details.append(f"FCF mostly negative — this is not a cash-generating business")

    # 2. FCF growth trajectory
    if len(fcf_values) >= 5:
        recent_fcf = sum(fcf_values[:3]) / 3
        older_fcf = sum(fcf_values[-3:]) / 3
        if older_fcf > 0 and recent_fcf > older_fcf:
            growth = (recent_fcf / older_fcf) - 1
            if growth > 0.50:
                score += 3
                details.append(f"Strong FCF growth: {growth:.0%} improvement over period")
            elif growth > 0.20:
                score += 2
                details.append(f"Good FCF growth: {growth:.0%} improvement")
            else:
                score += 1
                details.append(f"Modest FCF growth: {growth:.0%}")
        elif older_fcf > 0:
            details.append("FCF declining — concerning for long-term owner earnings")

    # 3. Valuation: FCF yield (Duan's preferred lens — "is this a reasonable price?")
    normalized_fcf = sum(fcf_values[:min(5, len(fcf_values))]) / min(5, len(fcf_values))
    fcf_yield = None
    if market_cap and market_cap > 0 and normalized_fcf > 0:
        fcf_yield = normalized_fcf / market_cap
        if fcf_yield > 0.07:
            score += 3
            details.append(f"Excellent FCF yield: {fcf_yield:.1%} — very reasonable price for quality")
        elif fcf_yield > 0.04:
            score += 2
            details.append(f"Good FCF yield: {fcf_yield:.1%}")
        elif fcf_yield > 0.02:
            score += 1
            details.append(f"Fair FCF yield: {fcf_yield:.1%} — priced for quality")
        else:
            details.append(f"Low FCF yield: {fcf_yield:.1%} — priced for perfection")

    final_score = min(10, score * 10 / 9)
    return {
        "score": final_score,
        "details": "; ".join(details),
        "fcf_yield": fcf_yield,
        "normalized_fcf": normalized_fcf if normalized_fcf > 0 else None,
    }


def evaluate_management_character(financial_line_items: list, insider_trades: list) -> dict:
    """
    Management '本分' evaluation — Duan's key qualitative filter.

    He says the first question about management is: are they 本分?
    Do they stay in their lane, not enrich themselves at shareholder expense,
    and speak honestly about mistakes?

    Quantitative proxies for 本分 management:
    - FCF conversion (do earnings turn into real cash?)
    - Share dilution (are they enriching themselves?)
    - Debt conservatism (are they reckless with the balance sheet?)
    - Insider buying (are they betting their own money?)
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": 0, "details": "No data to evaluate management"}

    # 1. Cash conversion — 本分 management delivers real earnings, not accounting tricks
    fcf_values = [
        item.free_cash_flow for item in financial_line_items
        if hasattr(item, "free_cash_flow") and item.free_cash_flow is not None
    ]
    ni_values = [
        item.net_income for item in financial_line_items
        if hasattr(item, "net_income") and item.net_income is not None
    ]
    if fcf_values and ni_values and len(fcf_values) == len(ni_values):
        ratios = [
            fcf_values[i] / ni_values[i]
            for i in range(len(fcf_values))
            if ni_values[i] and ni_values[i] > 0
        ]
        if ratios:
            avg = sum(ratios) / len(ratios)
            if avg > 1.0:
                score += 3
                details.append(f"Excellent cash conversion: FCF/NI = {avg:.2f} — management delivers real earnings")
            elif avg > 0.8:
                score += 2
                details.append(f"Good cash conversion: FCF/NI = {avg:.2f}")
            elif avg > 0.6:
                score += 1
                details.append(f"Adequate cash conversion: FCF/NI = {avg:.2f}")
            else:
                details.append(f"Poor cash conversion: FCF/NI = {avg:.2f} — accounting quality concern")

    # 2. Share count discipline — not diluting shareholders
    shares = [
        item.outstanding_shares for item in financial_line_items
        if hasattr(item, "outstanding_shares") and item.outstanding_shares is not None
    ]
    if shares and len(shares) >= 3:
        if shares[0] < shares[-1] * 0.95:  # buybacks
            score += 2
            details.append("Returning capital via buybacks — shareholder-aligned management")
        elif shares[0] <= shares[-1] * 1.03:  # stable
            score += 1
            details.append("Stable share count — no significant dilution")
        else:
            details.append(f"Share count increased {(shares[0]/shares[-1]-1):.0%} — diluting shareholders")

    # 3. Debt conservatism
    debt = [
        item.total_debt for item in financial_line_items
        if hasattr(item, "total_debt") and item.total_debt is not None
    ]
    equity = [
        item.shareholders_equity for item in financial_line_items
        if hasattr(item, "shareholders_equity") and item.shareholders_equity is not None
    ]
    if debt and equity:
        de = debt[0] / equity[0] if equity[0] > 0 else float("inf")
        if de < 0.2:
            score += 3
            details.append(f"Zero-debt discipline: D/E {de:.2f} — Duan's hallmark")
        elif de < 0.5:
            score += 2
            details.append(f"Conservative leverage: D/E {de:.2f}")
        elif de < 1.0:
            score += 1
            details.append(f"Moderate leverage: D/E {de:.2f}")
        else:
            details.append(f"High leverage: D/E {de:.2f} — misaligned with Duan's principles")

    # 4. Insider activity
    if insider_trades:
        buys = sum(1 for t in insider_trades
                   if getattr(t, "transaction_type", None)
                   and t.transaction_type.lower() in ["buy", "purchase"])
        sells = sum(1 for t in insider_trades
                    if getattr(t, "transaction_type", None)
                    and t.transaction_type.lower() in ["sell", "sale"])
        total = buys + sells
        if total > 0:
            buy_ratio = buys / total
            if buy_ratio > 0.6:
                score += 2
                details.append(f"Insiders buying: {buy_ratio:.0%} of trades are purchases — skin in the game")
            elif buy_ratio > 0.3:
                score += 1
                details.append(f"Mixed insider activity: {buy_ratio:.0%} purchases")
            elif sells > 5:
                details.append("Heavy insider selling — management exiting")

    final_score = min(10, score * 10 / 10)
    return {"score": final_score, "details": "; ".join(details)}


def assess_ten_year_predictability(financial_line_items: list) -> dict:
    """
    10-year business predictability — Duan's investment horizon is a decade.

    He asks: "Can I roughly predict what this company will earn 10 years from now?"
    If the answer requires a crystal ball, he passes.

    Duan only invests in businesses he can predict without modeling — the answer
    should be "obvious" if the business is truly great and defensive.
    """
    score = 0
    details = []

    if not financial_line_items or len(financial_line_items) < 5:
        return {"score": 0, "details": "Need 5+ years of history to assess 10-year predictability"}

    # 1. Revenue stability and growth consistency
    revenues = [
        item.revenue for item in financial_line_items
        if hasattr(item, "revenue") and item.revenue is not None
    ]
    if revenues and len(revenues) >= 5:
        growth_rates = [
            (revenues[i] / revenues[i+1]) - 1
            for i in range(len(revenues)-1)
            if revenues[i+1] and revenues[i+1] > 0
        ]
        if growth_rates:
            avg_growth = sum(growth_rates) / len(growth_rates)
            volatility = sum(abs(r - avg_growth) for r in growth_rates) / len(growth_rates)
            if avg_growth > 0.05 and volatility < 0.08:
                score += 3
                details.append(f"Highly predictable growth: {avg_growth:.1%} avg, very low volatility")
            elif avg_growth > 0 and volatility < 0.15:
                score += 2
                details.append(f"Reasonably predictable: {avg_growth:.1%} avg growth, moderate consistency")
            elif avg_growth > 0:
                score += 1
                details.append(f"Growing but volatile: {avg_growth:.1%} avg with high variance")
            else:
                details.append(f"Declining or erratic revenue — cannot predict 10 years out")

    # 2. Operating margin consistency — the signature of a durable moat
    op_margins = [
        item.operating_margin for item in financial_line_items
        if hasattr(item, "operating_margin") and item.operating_margin is not None
    ]
    if op_margins and len(op_margins) >= 5:
        avg_margin = sum(op_margins) / len(op_margins)
        margin_vol = sum(abs(m - avg_margin) for m in op_margins) / len(op_margins)
        if margin_vol < 0.03 and avg_margin > 0.15:
            score += 3
            details.append(f"Rock-solid margins: {avg_margin:.1%} avg with minimal variance — predictable moat")
        elif margin_vol < 0.06:
            score += 2
            details.append(f"Stable margins: {avg_margin:.1%} avg")
        elif avg_margin > 0.10:
            score += 1
            details.append(f"Reasonable but volatile margins: {avg_margin:.1%} avg")
        else:
            details.append(f"Thin or volatile margins — unpredictable industry structure")

    # 3. Operating income streak — has it ever lost money on operations?
    op_income = [
        item.operating_income for item in financial_line_items
        if hasattr(item, "operating_income") and item.operating_income is not None
    ]
    if op_income and len(op_income) >= 5:
        always_positive = all(x > 0 for x in op_income)
        mostly_positive = sum(1 for x in op_income if x > 0) / len(op_income)
        if always_positive:
            score += 2
            details.append("Operating income positive every year — model durability confirmed")
        elif mostly_positive >= 0.8:
            score += 1
            details.append(f"Operating income mostly positive ({mostly_positive:.0%} of years)")
        else:
            details.append("Significant operating losses in history — unpredictable")

    final_score = min(10, score * 10 / 8)
    return {"score": final_score, "details": "; ".join(details)}


# ---------------------------------------------------------------------------
# OUTPUT GENERATION
# ---------------------------------------------------------------------------

def _r(x, n=3):
    try:
        return round(float(x), n)
    except Exception:
        return None


def make_duan_facts_bundle(analysis: dict) -> dict:
    benfan = analysis.get("benfan_analysis") or {}
    stop = analysis.get("stop_doing_analysis") or {}
    fcf = analysis.get("fcf_analysis") or {}
    mgmt = analysis.get("management_analysis") or {}
    pred = analysis.get("predictability_analysis") or {}

    flags = {
        "benfan_strong": (_r(benfan.get("score"), 2) or 0) >= 7,
        "no_stop_doing_violations": len(stop.get("violations") or []) == 0,
        "strong_fcf": (_r(fcf.get("score"), 2) or 0) >= 7,
        "management_benfan": (_r(mgmt.get("score"), 2) or 0) >= 7,
        "ten_year_predictable": (_r(pred.get("score"), 2) or 0) >= 7,
        "fcf_yield_attractive": (_r(fcf.get("fcf_yield"), 4) or 0) >= 0.04,
        "stop_doing_penalty": _r(stop.get("penalty"), 2),
    }

    return {
        "pre_signal": analysis.get("signal"),
        "total_score": _r(analysis.get("score"), 2),
        "benfan_score": _r(benfan.get("score"), 2),
        "fcf_score": _r(fcf.get("score"), 2),
        "management_score": _r(mgmt.get("score"), 2),
        "predictability_score": _r(pred.get("score"), 2),
        "stop_doing_penalty": _r(stop.get("penalty"), 2),
        "fcf_yield": _r(fcf.get("fcf_yield"), 4),
        "normalized_fcf": _r(fcf.get("normalized_fcf"), 0),
        "flags": flags,
        "notes": {
            "benfan": (benfan.get("details") or "")[:150],
            "stop_doing": (stop.get("details") or "")[:150],
            "fcf": (fcf.get("details") or "")[:150],
            "management": (mgmt.get("details") or "")[:150],
            "predictability": (pred.get("details") or "")[:150],
        },
    }


def compute_confidence(analysis: dict, signal: str) -> int:
    benfan = float((analysis.get("benfan_analysis") or {}).get("score") or 0)
    fcf = float((analysis.get("fcf_analysis") or {}).get("score") or 0)
    mgmt = float((analysis.get("management_analysis") or {}).get("score") or 0)
    pred = float((analysis.get("predictability_analysis") or {}).get("score") or 0)
    penalty = float((analysis.get("stop_doing_analysis") or {}).get("penalty") or 0)

    # Duan's weighting: quality + FCF matter most
    quality_pct = 100 * (0.30 * benfan + 0.30 * fcf + 0.20 * mgmt + 0.20 * pred) / 10.0
    quality_pct *= (1 - penalty)

    if signal == "bullish":
        lower, upper = 55, 100
    elif signal == "bearish":
        lower, upper = 10, 49
    else:
        lower, upper = 45, 69

    return max(10, min(100, int(round(max(lower, min(upper, quality_pct))))))


def generate_duan_output(
    ticker: str,
    analysis_data: dict,
    state: AgentState,
    agent_id: str,
    confidence_hint: int,
) -> DuanYongpingSignal:
    facts_bundle = make_duan_facts_bundle(analysis_data)

    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are Duan Yongping (段永平), the Chinese entrepreneur-investor. "
         "You founded Bubu Hi-Tech (步步高), co-created OPPO and vivo, and are a major "
         "investor in Apple, Tencent, and Pinduoduo. You live in Palo Alto and manage a "
         "personal portfolio with Warren Buffett as your primary intellectual mentor. "
         "Your principles: '做对的事，把事情做对', '不为清单', '本分', zero leverage, FCF over EPS. "
         "You refuse to predict short-term prices. You speak plainly and directly. "
         "Decide bullish, bearish, or neutral using ONLY the provided facts. "
         "Return JSON only. Keep reasoning under 150 characters. Speak as Duan would — "
         "grounded, honest, focused on business quality not price speculation. "
         "Use the provided confidence exactly."),
        ("human",
         "Ticker: {ticker}\n"
         "Facts:\n{facts}\n"
         "Confidence: {confidence}\n"
         "Return exactly:\n"
         "{{\n"
         '  "signal": "bullish" | "bearish" | "neutral",\n'
         f'  "confidence": {confidence_hint},\n'
         '  "reasoning": "short justification in Duan\'s voice"\n'
         "}}")
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "facts": json.dumps(facts_bundle, separators=(",", ":"), ensure_ascii=False),
        "confidence": confidence_hint,
    })

    def _default():
        return DuanYongpingSignal(
            signal="neutral",
            confidence=confidence_hint,
            reasoning="Insufficient data — I don't invest in what I don't understand",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=DuanYongpingSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )
