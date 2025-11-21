"""
Data Manager for Backtesting Module.
Handles fetching historical data from Binance and caching it in a local SQLite database.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import os

from src.binance_client import BinanceClient
from src.logger import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    Manages historical market data for backtesting.
    Caches data in SQLite to minimize API calls.
    """

    def __init__(self, db_path: str, binance_client: BinanceClient):
        """
        Initialize DataManager.

        Args:
            db_path: Path to SQLite database file
            binance_client: Instance of BinanceClient for fetching fresh data
        """
        self.db_path = db_path
        self.client = binance_client
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    symbol TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON klines (symbol, timestamp)")

    def get_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime, 
        interval: str = "5m"
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        Checks DB first, fetches missing data from Binance, caches it, and returns the full range.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            start_time: Start datetime
            end_time: End datetime
            interval: Kline interval (only '5m' supported for now as per DB schema simplicity)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # 1. Check what we have in DB
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        df_db = self._fetch_from_db(symbol, start_ts, end_ts)
        
        # 2. Identify missing ranges
        # This is a simplified approach: if we have gaps, we might just fetch the whole range 
        # or specific chunks. For robustness, let's fetch everything if DB is empty 
        # or if we are missing significant data at edges.
        
        # A simple robust strategy: 
        # If DB returns nothing, fetch all.
        # If DB returns data, check min/max timestamps. Fetch missing head/tail.
        # Gaps in the middle are harder to detect without scanning, but we assume contiguous writes.
        
        if df_db.empty:
            logger.info(f"No cached data for {symbol}. Fetching full range from Binance...")
            df_new = self.client.get_historical_klines(symbol, interval, start_time, end_time)
            if not df_new.empty:
                self._save_to_db(symbol, df_new)
            return df_new
        
        # Check edges
        db_min_ts = df_db["timestamp"].min().timestamp() * 1000
        db_max_ts = df_db["timestamp"].max().timestamp() * 1000
        
        # Fetch missing head (before DB data)
        if start_ts < db_min_ts:
            missing_start = start_time
            missing_end = datetime.fromtimestamp(db_min_ts / 1000)
            logger.info(f"Fetching missing head for {symbol}: {missing_start} to {missing_end}")
            df_head = self.client.get_historical_klines(symbol, interval, missing_start, missing_end)
            if not df_head.empty:
                self._save_to_db(symbol, df_head)
                
        # Fetch missing tail (after DB data)
        if end_ts > db_max_ts:
            missing_start = datetime.fromtimestamp(db_max_ts / 1000)
            missing_end = end_time
            logger.info(f"Fetching missing tail for {symbol}: {missing_start} to {missing_end}")
            df_tail = self.client.get_historical_klines(symbol, interval, missing_start, missing_end)
            if not df_tail.empty:
                self._save_to_db(symbol, df_tail)
        
        # 3. Return combined data from DB
        return self._fetch_from_db(symbol, start_ts, end_ts)

    def _fetch_from_db(self, symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        """Fetch data from SQLite."""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM klines
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=(symbol, start_ts, end_ts))
            
        if not df.empty:
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def _save_to_db(self, symbol: str, df: pd.DataFrame):
        """Save DataFrame to SQLite."""
        if df.empty:
            return
            
        # Prepare data for insertion
        data = df.copy()
        data["symbol"] = symbol
        # Ensure timestamp is integer milliseconds
        data["timestamp"] = data["timestamp"].astype(int) // 10**6 
        
        records = data[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].to_records(index=False)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO klines (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
            logger.info(f"Cached {len(records)} records for {symbol}")
