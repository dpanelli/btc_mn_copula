"""
Statistics module for backtesting.
Calculates performance metrics from equity curves.
"""

import pandas as pd
import numpy as np
from typing import Dict

def calculate_stats(equity_curve: pd.DataFrame) -> Dict:
    """
    Calculate performance statistics.
    
    Args:
        equity_curve: DataFrame with 'timestamp' and 'equity' columns.
        
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
    
    # Max Time in Drawdown
    # TODO: Calculate duration of drawdowns
    
    return {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{volatility:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Final Equity": f"${final_equity:,.2f}"
    }
