"""
Main script to run backtests.
"""

import argparse
import sys
import os
from datetime import datetime
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.config import get_config
from src.binance_client import BinanceClient
from src.backtesting.data_manager import DataManager
from src.backtesting.engine import BacktestEngine
from src.backtesting.stats import calculate_stats
from src.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Run Backtest")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="backtest_results", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup output dir
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logger
    logger = setup_logger("backtest", f"{args.output}/backtest.log")
    
    # Load config
    config = get_config()
    
    # Initialize components
    binance_client = BinanceClient(
        api_key=config.binance.api_key,
        api_secret=config.binance.api_secret
    )
    
    data_manager = DataManager(
        db_path="data/market_data.db",
        binance_client=binance_client
    )
    
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    
    # Run Engine
    engine = BacktestEngine(
        data_manager=data_manager,
        config=config,
        start_date=start_date,
        end_date=end_date
    )
    
    engine.run()
    
    # Generate Report
    equity_df = pd.DataFrame(engine.equity_curve)
    stats = calculate_stats(equity_df)
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS")
    print("="*40)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("="*40)
    
    # Save results
    equity_df.to_csv(f"{args.output}/equity.csv", index=False)
    
    # Plot equity
    if not equity_df.empty and HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df["timestamp"], equity_df["equity"])
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(f"{args.output}/equity.png")
        print(f"Equity chart saved to {args.output}/equity.png")
    elif not HAS_MATPLOTLIB:
        print("Matplotlib not found, skipping chart generation.")

if __name__ == "__main__":
    main()
