import unittest
from unittest.mock import patch, MagicMock
from src.tools.mx_adapter import warm_financial_metrics_cache
from src.data.cache import get_cache

class TestMXAdapterBatch(unittest.TestCase):
    def setUp(self):
        self.cache = get_cache()
        # Clear cache before each test
        self.cache._financial_metrics_cache = {}
        self.cache._line_items_cache = {}

    @patch("src.tools.mx_adapter._mx_query_tables")
    def test_warm_financial_metrics_cache_with_ratios(self, mock_query):
        # Mock data representing a batch response from MX with PE/PB/PS/DivYield and core metrics
        mock_table1 = {
            "sheet_name": "贵州茅台(600519.SH)",
            "rows": [
                {
                    "date": "2023-12-31",
                    "净利润": "747亿元",
                    "营业收入": "1500亿元",
                    "资产总计": "2000亿元",
                    "流动负债合计": "200亿元",
                    "流动资产合计": "600亿元",
                    "归母净利润同比增长": "15%",
                }
            ],
            "fieldnames": ["date", "净利润", "营业收入", "资产总计", "流动负债合计", "流动资产合计", "归母净利润同比增长"]
        }
        
        mock_table2 = {
            "sheet_name": "贵州茅台(600519.SH)",
            "rows": [
                {
                    "date": "2023-12-31",
                    "市盈率TTM": "25.5",
                    "市净率": "7.8",
                    "市销率": "12.3",
                    "股息率": "2.5%"
                }
            ],
            "fieldnames": ["date", "市盈率TTM", "市净率", "市销率", "股息率"]
        }
        
        # mock_query is called twice in warm_financial_metrics_cache
        mock_query.side_effect = [
            ([mock_table1], ["贵州茅台(600519.SH)"], 1, None),
            ([mock_table2], ["贵州茅台(600519.SH)"], 1, None)
        ]

        tickers = ["600519.SH"]
        end_date = "2024-03-01"
        warm_financial_metrics_cache(tickers, end_date, period="ttm", limit=1)

        # Verify cache
        cache_key = f"600519.SH_ttm_{end_date}_1"
        cached_data = self.cache.get_financial_metrics(cache_key)
        
        self.assertIsNotNone(cached_data)
        self.assertEqual(len(cached_data), 1)
        
        metrics = cached_data[0]
        # Base Ratios
        self.assertEqual(metrics.get("price_to_earnings_ratio"), 25.5)
        self.assertEqual(metrics.get("price_to_book_ratio"), 7.8)
        self.assertEqual(metrics.get("price_to_sales_ratio"), 12.3)
        self.assertEqual(metrics.get("free_cash_flow_yield"), 0.025) # Div yield mapped to FCF yield
        
        # Computed Ratios
        self.assertEqual(metrics.get("current_ratio"), 3.0) # 600 / 200
        self.assertEqual(metrics.get("asset_turnover"), 0.75) # 1500 / 2000
        self.assertEqual(metrics.get("peg_ratio"), 1.7) # 25.5 / (0.15 * 100)
        self.assertEqual(metrics.get("earnings_growth"), 0.15)

    @patch("src.tools.mx_adapter._mx_query_tables")
    def test_warm_line_items_cache(self, mock_query):
        from src.tools.mx_adapter import warm_line_items_cache
        
        # Mock data for line items including EBIT, EBITDA (per share), and shares
        mock_table = {
            "sheet_name": "贵州茅台(600519.SH)",
            "rows": [
                {
                    "date": "2023-12-31",
                    "净利润": "747.3亿元",
                    "研发费用": "1.5亿元",
                    "每股收益EPS(基本)": "59.49",
                    "流动资产合计": "1000亿元",
                    "流动负债合计": "200亿元",
                    "总股本": "12.56亿",
                    "每股息税折旧摊销前利润EBITDAPS": "82.5",
                    "息税前利润(TTM)": "1038亿元"
                }
            ],
            "fieldnames": ["date", "净利润", "研发费用", "每股收益EPS(基本)", "流动资产合计", "流动负债合计", "总股本", "每股息税折旧摊销前利润EBITDAPS", "息税前利润(TTM)"]
        }
        mock_query.return_value = ([mock_table], ["贵州茅台(600519.SH)"], 1, None)

        tickers = ["600519.SH"]
        line_items = ["net_income", "research_and_development", "earnings_per_share", "current_assets", "current_liabilities", "ebitda", "ebit"]
        end_date = "2024-03-01"
        warm_line_items_cache(tickers, line_items, end_date, limit=1)

        # Verify cache
        cache_key = f"600519.SH_ttm_{end_date}_1"
        cached_data = self.cache.get_line_items(cache_key)
        
        self.assertIsNotNone(cached_data)
        self.assertEqual(len(cached_data), 1)
        
        item = cached_data[0]
        self.assertEqual(item.get("net_income"), 74730000000.0)
        self.assertEqual(item.get("research_and_development"), 150000000.0)
        self.assertEqual(item.get("earnings_per_share"), 59.49)
        self.assertEqual(item.get("current_assets"), 100000000000.0)
        self.assertEqual(item.get("current_liabilities"), 20000000000.0)
        
        # Absolute EBIT
        self.assertEqual(item.get("ebit"), 103800000000.0)
        # EBITDA per share * Outstanding Shares = 82.5 * 1.256B
        self.assertEqual(item.get("ebitda"), 82.5 * 1256000000.0)

if __name__ == '__main__':
    unittest.main()
