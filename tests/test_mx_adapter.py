import unittest
from unittest.mock import patch, MagicMock
from src.tools.mx_adapter import warm_financial_metrics_cache
from src.data.cache import get_cache

class TestMXAdapterBatch(unittest.TestCase):
    def setUp(self):
        self.cache = get_cache()
        # Clear cache before each test
        self.cache._financial_metrics_cache = {}

    @patch("src.tools.mx_adapter._mx_query_tables")
    def test_warm_financial_metrics_cache_with_ratios(self, mock_query):
        # Mock data representing a batch response from MX with PE/PB/PS/DivYield
        mock_table = {
            "sheet_name": "贵州茅台(600519.SH)",
            "rows": [
                {
                    "date": "2023-12-31",
                    "净利润": "747亿元",
                    "市盈率TTM": "25.5",
                    "市净率": "7.8",
                    "市销率": "12.3",
                    "股息率": "2.5%"
                }
            ],
            "fieldnames": ["date", "净利润", "市盈率TTM", "市净率", "市销率", "股息率"]
        }
        mock_query.return_value = ([mock_table], ["贵州茅台(600519.SH)"], 1, None)

        tickers = ["600519.SH"]
        end_date = "2024-03-01"
        warm_financial_metrics_cache(tickers, end_date, period="ttm", limit=1)

        # Verify cache
        cache_key = f"600519.SH_ttm_{end_date}_1"
        cached_data = self.cache.get_financial_metrics(cache_key)
        
        self.assertIsNotNone(cached_data)
        self.assertEqual(len(cached_data), 1)
        
        metrics = cached_data[0]
        # Current implementation might not have these fields yet, so this should fail
        self.assertEqual(metrics.get("price_to_earnings_ratio"), 25.5)
        self.assertEqual(metrics.get("price_to_book_ratio"), 7.8)
        self.assertEqual(metrics.get("price_to_sales_ratio"), 12.3)

    @patch("src.tools.mx_adapter._mx_query_tables")
    def test_warm_line_items_cache(self, mock_query):
        from src.tools.mx_adapter import warm_line_items_cache
        
        # Mock data for line items (Net Income and R&D and EPS)
        mock_table = {
            "sheet_name": "贵州茅台(600519.SH)",
            "rows": [
                {
                    "date": "2023-12-31",
                    "净利润": "747.3亿元",
                    "研发费用": "1.5亿元",
                    "每股收益EPS(基本)": "59.49"
                }
            ],
            "fieldnames": ["date", "净利润", "研发费用", "每股收益EPS(基本)"]
        }
        mock_query.return_value = ([mock_table], ["贵州茅台(600519.SH)"], 1, None)

        tickers = ["600519.SH"]
        line_items = ["net_income", "research_and_development", "earnings_per_share"]
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
