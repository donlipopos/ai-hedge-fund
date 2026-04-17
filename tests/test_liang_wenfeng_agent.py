import json
from types import SimpleNamespace

from src.agents.liang_wenfeng import (
    analyze_quant_regime,
    compute_confidence,
    liang_wenfeng_agent,
)
from src.data.models import Price
from src.utils.analysts import ANALYST_CONFIG, get_agents_list, get_analyst_nodes


def _metric(**overrides):
    base = {
        "ticker": "AAPL",
        "report_period": "2025-12-31",
        "period": "annual",
        "currency": "USD",
        "market_cap": 1_000_000_000.0,
        "enterprise_value": None,
        "price_to_earnings_ratio": None,
        "price_to_book_ratio": None,
        "price_to_sales_ratio": 3.5,
        "enterprise_value_to_ebitda_ratio": None,
        "enterprise_value_to_revenue_ratio": None,
        "free_cash_flow_yield": None,
        "peg_ratio": 1.2,
        "gross_margin": 0.58,
        "operating_margin": 0.22,
        "net_margin": None,
        "return_on_equity": 0.20,
        "return_on_assets": None,
        "return_on_invested_capital": 0.18,
        "asset_turnover": None,
        "inventory_turnover": None,
        "receivables_turnover": None,
        "days_sales_outstanding": None,
        "operating_cycle": None,
        "working_capital_turnover": None,
        "current_ratio": None,
        "quick_ratio": None,
        "cash_ratio": None,
        "operating_cash_flow_ratio": None,
        "debt_to_equity": 0.15,
        "debt_to_assets": None,
        "interest_coverage": None,
        "revenue_growth": 0.24,
        "earnings_growth": None,
        "book_value_growth": None,
        "earnings_per_share_growth": None,
        "free_cash_flow_growth": None,
        "operating_income_growth": None,
        "ebitda_growth": None,
        "payout_ratio": None,
        "earnings_per_share": None,
        "book_value_per_share": None,
        "free_cash_flow_per_share": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _line_item(**overrides):
    base = {
        "ticker": "AAPL",
        "report_period": "2025-12-31",
        "period": "annual",
        "currency": "USD",
        "revenue": 1000.0,
        "free_cash_flow": 180.0,
        "capital_expenditure": -35.0,
        "gross_margin": 0.58,
        "operating_margin": 0.22,
        "net_income": 150.0,
        "research_and_development": 140.0,
        "total_debt": 80.0,
        "shareholders_equity": 520.0,
        "cash_and_equivalents": 150.0,
        "outstanding_shares": 100.0,
        "operating_income": 210.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _state():
    return {
        "messages": [],
        "data": {
            "tickers": ["300750.SZ"],
            "portfolio": {},
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "analyst_signals": {},
        },
        "metadata": {
            "show_reasoning": False,
            "model_name": "gpt-4.1",
            "model_provider": "OpenAI",
        },
    }


def test_registry_exposes_liang_wenfeng():
    assert "liang_wenfeng" in ANALYST_CONFIG
    assert "liang_wenfeng" in get_analyst_nodes()

    agents = get_agents_list()
    liang = next(agent for agent in agents if agent["key"] == "liang_wenfeng")
    assert liang["display_name"] == "Liang Wenfeng"
    assert "China AI Founder-Quant" in liang["description"]


def test_analyze_quant_regime_scores_trending_series_higher():
    bullish_prices = [
        Price(open=float(i), close=float(i), high=float(i), low=float(i), volume=1000, time=f"2025-01-{(i % 28) + 1:02d}")
        for i in range(100, 180)
    ]
    bearish_prices = [
        Price(open=float(i), close=float(i), high=float(i), low=float(i), volume=1000, time=f"2025-02-{(idx % 28) + 1:02d}")
        for idx, i in enumerate(range(180, 100, -1))
    ]

    bullish = analyze_quant_regime(bullish_prices)
    bearish = analyze_quant_regime(bearish_prices)

    assert bullish["score"] > bearish["score"]
    assert bullish["score"] >= 7
    assert bearish["score"] <= 4


def test_compute_confidence_penalizes_sparse_analysis():
    dense = {
        "signal": "bullish",
        "score": 8.0,
        "innovation_analysis": {"details": "good"},
        "capital_efficiency_analysis": {"details": "good"},
        "founder_quality_analysis": {"details": "good"},
        "quant_regime_analysis": {"details": "good"},
        "china_resilience_analysis": {"details": "good"},
        "valuation_analysis": {"details": "good"},
    }
    sparse = {
        "signal": "neutral",
        "score": 5.5,
        "innovation_analysis": {"details": "good"},
        "capital_efficiency_analysis": {"details": ""},
        "founder_quality_analysis": {"details": ""},
        "quant_regime_analysis": {"details": "Insufficient price history; quant regime defaults to cautious neutral"},
        "china_resilience_analysis": {"details": ""},
        "valuation_analysis": {"details": ""},
    }

    assert compute_confidence(dense) > compute_confidence(sparse)


def test_liang_wenfeng_agent_emits_signal_with_monkeypatched_data(monkeypatch):
    metrics = [_metric()]
    line_items = [
        _line_item(report_period="2025-12-31", revenue=1000.0, free_cash_flow=180.0, outstanding_shares=100.0),
        _line_item(report_period="2024-12-31", revenue=860.0, free_cash_flow=150.0, outstanding_shares=101.0),
        _line_item(report_period="2023-12-31", revenue=720.0, free_cash_flow=110.0, outstanding_shares=102.0),
    ]
    trades = [SimpleNamespace(transaction_type="Purchase"), SimpleNamespace(transaction_type="Buy")]
    news = [
        SimpleNamespace(title="China AI platform launches efficient open source model", source="Tech"),
        SimpleNamespace(title="Domestic supply chain and developer adoption accelerate", source="News"),
        SimpleNamespace(title="Semiconductor and AI demand remains strong in China", source="Wire"),
    ]
    prices = [
        Price(open=float(i), close=float(i), high=float(i), low=float(i), volume=1000, time=f"2025-03-{(idx % 28) + 1:02d}")
        for idx, i in enumerate(range(100, 180))
    ]

    monkeypatch.setattr("src.agents.liang_wenfeng.get_financial_metrics", lambda *a, **k: metrics)
    monkeypatch.setattr("src.agents.liang_wenfeng.search_line_items", lambda *a, **k: line_items)
    monkeypatch.setattr("src.agents.liang_wenfeng.get_market_cap", lambda *a, **k: 2_500.0)
    monkeypatch.setattr("src.agents.liang_wenfeng.get_insider_trades", lambda *a, **k: trades)
    monkeypatch.setattr("src.agents.liang_wenfeng.get_company_news", lambda *a, **k: news)
    monkeypatch.setattr("src.agents.liang_wenfeng.get_prices", lambda *a, **k: prices)
    monkeypatch.setattr(
        "src.agents.liang_wenfeng.call_llm",
        lambda **kwargs: kwargs["default_factory"](),
    )

    result = liang_wenfeng_agent(_state())
    payload = json.loads(result["messages"][0].content)
    signal = payload["300750.SZ"]

    assert signal["signal"] == "bullish"
    assert signal["confidence"] >= 60
    assert "score" in signal["reasoning"].lower()
