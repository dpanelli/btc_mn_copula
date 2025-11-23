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
    parser.add_argument("--initial-capital", type=float, default=10000.0, help="Initial capital (default: 10000)")
    parser.add_argument("--parallel", action="store_true", help="Run backtest in parallel (optimized)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup output dir
    os.makedirs(args.output, exist_ok=True)
    
    # Setup logger
    logger = setup_logger("backtest", f"{args.output}/backtest.log")
    
    # Load config
    config = get_config()
    
    # IMPORTANT: Use LIVE data for backtesting (historical data is public, no auth required)
    # Override testnet setting regardless of .env
    config.binance.testnet = False
    
    # Initialize components with LIVE client (no API keys needed for historical data)
    binance_client = BinanceClient(
        api_key="",  # Not needed for historical data
        api_secret="",  # Not needed for historical data
        testnet=False  # Use LIVE historical data
    )
    
    data_manager = DataManager(
        db_path="data/market_data.db",
        binance_client=binance_client
    )
    
    from datetime import timezone
    start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    
    # Run Engine
    engine = BacktestEngine(
        data_manager=data_manager,
        config=config,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.initial_capital,
        output_dir=args.output
    )
    
    if args.parallel:
        engine.run_parallel()
    else:
        engine.run()
    
    # Generate Report
    equity_df = pd.DataFrame(engine.equity_curve)
    stats = calculate_stats(equity_df, engine.trades)
    
    print("\n" + "="*40)
    print("BACKTEST RESULTS")
    print("="*40)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("="*40)
    
    # Save results
    equity_df.to_csv(f"{args.output}/equity.csv", index=False)
    
    # Save trades to CSV
    if engine.trades:
        trades_data = []
        for trade in engine.trades:
            # Calculate leg PnLs
            if trade.side == "LONG_S1_SHORT_S2":
                # LONG S1 SHORT S2 = BUY ALT1, SELL ALT2
                # Long Side: ALT1
                # Short Side: ALT2
                alt1, alt2 = trade.pair.split("-")
                long_side = alt1
                short_side = alt2
                
                pnl_long = trade.size_alt1 * (trade.exit_price_alt1 - trade.entry_price_alt1)
                pnl_short = -trade.size_alt2 * (trade.exit_price_alt2 - trade.entry_price_alt2)
                
            else:  # SHORT_S1_LONG_S2
                # SHORT S1 LONG S2 = SELL ALT1, BUY ALT2
                # Short Side: ALT1
                # Long Side: ALT2
                alt1, alt2 = trade.pair.split("-")
                short_side = alt1
                long_side = alt2
                
                pnl_short = -trade.size_alt1 * (trade.exit_price_alt1 - trade.entry_price_alt1)
                pnl_long = trade.size_alt2 * (trade.exit_price_alt2 - trade.entry_price_alt2)
                
            trades_data.append({
                "start_time": trade.entry_time,
                "end_time": trade.exit_time,
                "pair": trade.pair,
                "side": trade.side,
                "long_asset": long_side,
                "short_asset": short_side,
                "pnl_long": round(pnl_long, 4),
                "pnl_short": round(pnl_short, 4),
                "total_pnl": round(trade.pnl, 4)
            })
            
        trades_df = pd.DataFrame(trades_data)
        trades_df.to_csv(f"{args.output}/trades.csv", index=False)
        print(f"Trades saved to {args.output}/trades.csv")
    
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

    # Generate QuantStats HTML Report
    try:
        import quantstats as qs
        
        # Extend pandas to support quantstats
        qs.extend_pandas()
        
        print("\nGenerating QuantStats HTML report...")
        
        # Prepare returns series
        # QuantStats expects a pandas Series of returns with datetime index
        # User requested daily resampling to fix CAGR calculation
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        daily_equity = equity_df.set_index('timestamp')['equity'].resample('D').last().dropna()
        returns_series = daily_equity.pct_change().fillna(0)
        
        # Ensure tearsheet directory exists
        tearsheet_dir = f"{args.output}/tearsheet"
        os.makedirs(tearsheet_dir, exist_ok=True)
        
        # Generate report
        qs.reports.html(
            returns_series,
            output=f"{tearsheet_dir}/report.html",
            title="BTC-MN Copula Strategy Backtest",
            download_filename="report.html"
        )
        print(f"QuantStats report saved to {tearsheet_dir}/report.html")
        
    except ImportError:
        print("QuantStats not installed. Skipping HTML report generation.")
    except Exception as e:
        print(f"Error generating QuantStats report: {e}")

if __name__ == "__main__":
    main()
