import argparse
import sys
import os
from src.main import main as run_main
from src.backtesting.cli import main as backtest_main
from src.cli.ashare_pipeline import main as ashare_main

def main():
    # Global flags only (e.g. --json)
    # We use a separate parser to avoid consuming help for subcommands
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--json", action="store_true")
    
    # Handle help if no arguments
    if len(sys.argv) == 1:
        _print_help()
        sys.exit(0)
        
    # Check for help flags
    if "-h" in sys.argv or "--help" in sys.argv:
        if len(sys.argv) == 2:
            _print_help()
            sys.exit(0)
        # If it's 'fund run --help', we let the subcommand handle it below

    # Parse global flags
    global_args, remaining = global_parser.parse_known_args()
    
    if global_args.json:
        os.environ["OUTPUT_JSON"] = "1"
        os.environ["PROGRESS_SILENT"] = "1"
        # Remove --json from arguments so subparsers don't see it
        if "--json" in sys.argv:
            sys.argv.remove("--json")
    
    # Now the first argument in remaining should be the command
    if not remaining:
        _print_help()
        sys.exit(0)
        
    command = remaining[0]
    
    # Shift sys.argv to remove the command name so the subcommand sees only its own args
    # e.g. 'fund run --ticker AAPL' -> 'fund --ticker AAPL'
    sys.argv = [sys.argv[0]] + remaining[1:]
    
    if command == "run":
        run_main()
    elif command == "backtest":
        backtest_main()
    elif command in ("ashare", "ashare-pipeline"):
        ashare_main()
    else:
        print(f"Unknown command: {command}")
        _print_help()
        sys.exit(1)

def _print_help():
    print("""AI Hedge Fund Unified CLI

Usage:
  fund <command> [options]

Commands:
  run       Run standard hedge fund analysis (multi-agent workflow)
  backtest  Run historical simulation/backtester
  ashare    Run Chinese A-share pipeline (screening + analysis)

Global Options:
  --json    Output results as machine-readable JSON (silences progress bars)
  --help    Show this help message or help for a specific command

Examples:
  fund run --ticker AAPL,MSFT
  fund backtest --tickers NVDA --start-date 2024-01-01
  fund --json run --ticker AAPL
""")

if __name__ == "__main__":
    main()
