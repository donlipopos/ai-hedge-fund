import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import io
from src.cli.entrypoint import main

class TestCLI(unittest.TestCase):
    def setUp(self):
        # Clear environment variables that might be set by tests
        if "OUTPUT_JSON" in os.environ:
            del os.environ["OUTPUT_JSON"]
        if "PROGRESS_SILENT" in os.environ:
            del os.environ["PROGRESS_SILENT"]
        self.original_argv = sys.argv

    def tearDown(self):
        sys.argv = self.original_argv

    @patch('sys.exit')
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_help_no_args(self, mock_stdout, mock_exit):
        mock_exit.side_effect = SystemExit
        with patch('sys.argv', ['fund']):
            with self.assertRaises(SystemExit):
                main()
        
        output = mock_stdout.getvalue()
        self.assertIn("AI Hedge Fund Unified CLI", output)
        self.assertIn("Usage:", output)
        self.assertIn("Commands:", output)
        mock_exit.assert_called_once_with(0)

    @patch('src.cli.entrypoint.run_main')
    @patch('sys.exit')
    def test_run_command(self, mock_exit, mock_run_main):
        with patch('sys.argv', ['fund', 'run', '--ticker', 'AAPL']):
            main()
            # Verify sys.argv was shifted correctly for the subcommand
            self.assertEqual(sys.argv, ['fund', '--ticker', 'AAPL'])
        mock_run_main.assert_called_once()

    @patch('src.cli.entrypoint.backtest_main')
    @patch('sys.exit')
    def test_backtest_command(self, mock_exit, mock_backtest_main):
        with patch('sys.argv', ['fund', 'backtest', '--ticker', 'MSFT']):
            main()
            self.assertEqual(sys.argv, ['fund', '--ticker', 'MSFT'])
        mock_backtest_main.assert_called_once()

    @patch('src.cli.entrypoint.ashare_main')
    @patch('sys.exit')
    def test_ashare_command(self, mock_exit, mock_ashare_main):
        with patch('sys.argv', ['fund', 'ashare', '--criteria', 'ROE>10']):
            main()
            self.assertEqual(sys.argv, ['fund', '--criteria', 'ROE>10'])
        mock_ashare_main.assert_called_once()

    @patch('src.cli.entrypoint.run_main')
    @patch('sys.exit')
    def test_json_flag(self, mock_exit, mock_run_main):
        with patch('sys.argv', ['fund', '--json', 'run', '--ticker', 'AAPL']):
            main()
            self.assertEqual(os.environ.get("OUTPUT_JSON"), "1")
            self.assertEqual(os.environ.get("PROGRESS_SILENT"), "1")
            # Verify --json was removed and argv shifted
            self.assertEqual(sys.argv, ['fund', '--ticker', 'AAPL'])
        mock_run_main.assert_called_once()

    @patch('sys.exit', side_effect=SystemExit)
    @patch('sys.stdout', new_callable=io.StringIO)
    def test_unknown_command(self, mock_stdout, mock_exit):
        with patch('sys.argv', ['fund', 'invalid']):
            with self.assertRaises(SystemExit):
                main()
        
        output = mock_stdout.getvalue()
        self.assertIn("Unknown command: invalid", output)
        mock_exit.assert_called_once_with(1)

    @patch('src.cli.entrypoint.ashare_main')
    @patch('sys.exit')
    def test_ashare_pipeline_alias(self, mock_exit, mock_ashare_main):
        with patch('sys.argv', ['fund', 'ashare-pipeline', '--criteria', 'ROE>10']):
            main()
            self.assertEqual(sys.argv, ['fund', '--criteria', 'ROE>10'])
        mock_ashare_main.assert_called_once()

if __name__ == '__main__':
    unittest.main()
