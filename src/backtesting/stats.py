"""
Statistics module for backtesting.
Calculates performance metrics from equity curves.
"""

import pandas as pd
import numpy as np
from typing import Dict

def calculate_stats(equity_curve: pd.DataFrame, trades: list = None) -> Dict:
    """
    Calculate performance statistics.
    
    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity' columns.
        trades: List of Trade objects (dictionaries or objects)
        
    Returns:
        Dict of metrics.
    """
    if equity_curve.empty:
        return {}
        
    # Calculate returns
    equity_curve["returns"] = equity_curve["equity"].pct_change().fillna(0)
    
    # Total Return
    initial_equity = equity_curve["equity"].iloc[0]
    final_equity = equity_curve["equity"].iloc[-1]
    total_return = (final_equity - initial_equity) / initial_equity
    
    # Annualized Return (CAGR)
    days = (equity_curve["timestamp"].iloc[-1] - equity_curve["timestamp"].iloc[0]).days
    if days > 0:
        cagr = (final_equity / initial_equity) ** (365 / days) - 1
    else:
        cagr = 0
        
    # Volatility (Annualized)
    # Assuming 5-minute data (288 periods per day)
    volatility = equity_curve["returns"].std() * np.sqrt(288 * 365)
    
    # Sharpe Ratio (Risk Free Rate = 0)
    if volatility > 0:
        sharpe = cagr / volatility
    else:
        sharpe = 0
        
    # Sortino Ratio
    downside_returns = equity_curve["returns"][equity_curve["returns"] < 0]
    downside_vol = downside_returns.std() * np.sqrt(288 * 365)
    if downside_vol > 0:
        sortino = cagr / downside_vol
    else:
        sortino = 0
        
    # Max Drawdown
    equity_curve["peak"] = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["peak"]) / equity_curve["peak"]
    max_drawdown = equity_curve["drawdown"].min()
    
    # Max Absolute Drawdown
    equity_curve["abs_drawdown"] = equity_curve["peak"] - equity_curve["equity"]
    max_abs_drawdown = equity_curve["abs_drawdown"].max()
    
    # Drawdown Duration
    # Identify periods where equity is below peak
    is_in_drawdown = equity_curve["equity"] < equity_curve["peak"]
    
    max_drawdown_duration = pd.Timedelta(0)
    avg_drawdown_duration = pd.Timedelta(0)
    
    if is_in_drawdown.any():
        # Create groups for consecutive drawdown periods
        # (is_in_drawdown != is_in_drawdown.shift()).cumsum() creates a new group ID every time the state changes
        drawdown_groups = (is_in_drawdown != is_in_drawdown.shift()).cumsum()
        
        # Filter to only keep groups that are actually in drawdown
        drawdown_periods = equity_curve[is_in_drawdown].groupby(drawdown_groups)
        
        durations = []
        for _, group in drawdown_periods:
            start_time = group["timestamp"].iloc[0]
            end_time = group["timestamp"].iloc[-1]
            # Add the interval time (e.g. 5 mins) to the last candle to get full duration
            # Assuming 5 min interval if not detectable, but simple subtraction is a good approximation
            duration = end_time - start_time
            durations.append(duration)
            
        if durations:
            max_drawdown_duration = max(durations)
            avg_drawdown_duration = sum(durations, pd.Timedelta(0)) / len(durations)
    
    # Trade Statistics
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    total_trades = len(trades_df)
    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    max_win = 0.0
    max_loss = 0.0
    
    if not trades_df.empty and "pnl" in trades_df.columns:
        winning_trades = trades_df[trades_df["pnl"] > 0]
        losing_trades = trades_df[trades_df["pnl"] <= 0]
        
        if total_trades > 0:
            win_rate = len(winning_trades) / total_trades
            
        if not winning_trades.empty:
            avg_win = winning_trades["pnl"].mean()
            max_win = winning_trades["pnl"].max()
            
        if not losing_trades.empty:
            avg_loss = losing_trades["pnl"].mean()
            max_loss = losing_trades["pnl"].min()

    # Trade Duration Stats
    avg_trade_duration = pd.Timedelta(0)
    max_trade_duration = pd.Timedelta(0)

    if not trades_df.empty and "entry_time" in trades_df.columns and "exit_time" in trades_df.columns:
        # Ensure datetime
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        
        trades_df["duration"] = trades_df["exit_time"] - trades_df["entry_time"]
        
        if not trades_df["duration"].empty:
            avg_trade_duration = trades_df["duration"].mean()
            max_trade_duration = trades_df["duration"].max()

    return {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{volatility:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Max Abs Drawdown": f"${max_abs_drawdown:.2f}",
        "Avg Drawdown Time": str(avg_drawdown_duration).split('.')[0], # Remove microseconds
        "Max Drawdown Time": str(max_drawdown_duration).split('.')[0],
        "Final Equity": f"${final_equity:,.2f}",
        "Total Trades": total_trades,
        "Win Rate": f"{win_rate:.2%}",
        "Avg Win": f"${avg_win:.2f}",
        "Avg Loss": f"${avg_loss:.2f}",
        "Max Win": f"${max_win:.2f}",
        "Max Loss": f"${max_loss:.2f}",
        "Avg Trade Duration": str(avg_trade_duration).split('.')[0],
        "Max Trade Duration": str(max_trade_duration).split('.')[0]
    }
