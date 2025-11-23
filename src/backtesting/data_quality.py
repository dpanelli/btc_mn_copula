"""
Data Quality Checker for Backtesting.
Responsible for identifying data anomalies such as:
1. Static prices (zero variance) over extended periods.
2. Timestamp gaps.
3. Invalid values (NaN, zero, negative).
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from src.logger import get_logger

logger = get_logger(__name__)

class DataQualityChecker:
    """
    Analyzes market data for quality issues.
    """
    
    def __init__(self, static_price_threshold: int = 3):
        """
        Args:
            static_price_threshold: Number of consecutive candles with identical close price to consider 'static'.
                                    Default 3 (15 mins for 5m candles).
        """
        self.static_threshold = static_price_threshold

    def check_quality(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Run all quality checks and add an 'is_valid' column.
        
        Args:
            df: DataFrame with 'timestamp' and 'close' columns.
            symbol: Symbol name for logging.
            
        Returns:
            DataFrame with added 'is_valid' (bool) and 'quality_issue' (str) columns.
        """
        if df.empty:
            return df
            
        # Initialize
        df = df.copy()
        df['is_valid'] = True
        df['quality_issue'] = None
        
        # 1. Check for Static Prices
        # Calculate price difference
        df['price_diff'] = df['close'].diff()
        
        # Identify where diff is 0
        is_static = df['price_diff'] == 0.0
        
        # Group consecutive static candles
        # We assign a group ID that changes whenever is_static changes
        static_groups = (is_static != is_static.shift()).cumsum()
        
        # Filter for groups that are static (is_static=True)
        static_groups = static_groups[is_static]
        
        # Count size of each group
        group_counts = static_groups.value_counts()
        
        # Identify groups that exceed threshold
        bad_groups = group_counts[group_counts >= self.static_threshold].index
        
        # Mark bad rows
        # We need to map back to original index
        # Create a mask for all static rows
        mask_static_long = df.index.isin(static_groups[static_groups.isin(bad_groups)].index)
        
        if mask_static_long.any():
            count = mask_static_long.sum()
            logger.warning(f"[{symbol}] Found {count} candles with static prices (>{self.static_threshold} consecutive)")
            df.loc[mask_static_long, 'is_valid'] = False
            df.loc[mask_static_long, 'quality_issue'] = 'STATIC_PRICE'

        # 2. Check for NaNs (should be handled by loader, but double check)
        mask_nan = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
        if mask_nan.any():
            logger.warning(f"[{symbol}] Found {mask_nan.sum()} candles with NaN values")
            df.loc[mask_nan, 'is_valid'] = False
            df.loc[mask_nan, 'quality_issue'] = 'NAN_VALUES'
            
        # 3. Check for Zero/Negative Prices
        mask_zero = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if mask_zero.any():
            logger.warning(f"[{symbol}] Found {mask_zero.sum()} candles with zero/negative prices")
            df.loc[mask_zero, 'is_valid'] = False
            df.loc[mask_zero, 'quality_issue'] = 'INVALID_PRICE'

        # Cleanup temp columns
        df.drop(columns=['price_diff'], inplace=True)
        
        return df

    def detect_gaps(self, df: pd.DataFrame, interval_minutes: int = 5) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Detect timestamp gaps larger than the expected interval.
        """
        if df.empty:
            return []
            
        time_diff = df['timestamp'].diff()
        expected_diff = pd.Timedelta(minutes=interval_minutes)
        
        # Allow small tolerance (e.g. 1 second)
        gap_mask = time_diff > (expected_diff + pd.Timedelta(seconds=1))
        
        gaps = []
        if gap_mask.any():
            gap_indices = df.index[gap_mask]
            for idx in gap_indices:
                end_gap = df.loc[idx, 'timestamp']
                start_gap = df.loc[idx-1, 'timestamp'] # Previous candle
                gaps.append((start_gap, end_gap))
                
        return gaps
