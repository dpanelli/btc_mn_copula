"""
Data Manager for Backtesting Module.
Handles fetching historical data from Binance and caching it in a local SQLite database.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
import os

import time

from src.binance_client import BinanceClient
from src.backtesting.data_quality import DataQualityChecker
from src.backtesting.binance_vision_loader import BinanceVisionLoader
from src.logger import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    Manages historical market data for backtesting.
    Caches data in SQLite to minimize API calls.
    """

    def __init__(self, db_path: str, binance_client: Optional[BinanceClient] = None, read_only: bool = False):
        """
        Initialize DataManager.

        Args:
            db_path: Path to SQLite database file
            binance_client: Instance of BinanceClient for fetching fresh data (optional for read-only)
            read_only: If True, skip DB initialization (for worker processes)
        """
        self.db_path = db_path
        self.client = binance_client
        self.qc = DataQualityChecker()
        self.vision_loader = BinanceVisionLoader()
        self.read_only = read_only
        
        if not read_only:
            self._ensure_db()
    
    def seed_from_vision(self, symbols: List[str], start: datetime, end: datetime, interval: str = "5m"):
        """
        Seed the database using Binance Vision data (HTTP download).
        Optimized:
        1. Checks DB first (skips existing months).
        2. Downloads in PARALLEL.
        """
        import concurrent.futures
        
        logger.info(f"Seeding database from Binance Vision for {len(symbols)} symbols...")
        
        months = self.vision_loader.get_months_range(start, end)
        tasks = []
        
        # 1. Identify missing data
        logger.info("Checking existing data in DB...")
        for symbol in symbols:
            for year, month in months:
                # Check if we have data for this month
                # Simple check: Do we have any data for the 15th of the month? 
                # Or better: Count records. 
                # Fast check: Check if we have data for start/end of month.
                
                # Construct approximate start/end ms for this month
                m_start = datetime(int(year), int(month), 1, tzinfo=timezone.utc)
                if int(month) == 12:
                    m_end = datetime(int(year)+1, 1, 1, tzinfo=timezone.utc) - timedelta(milliseconds=1)
                else:
                    m_end = datetime(int(year), int(month)+1, 1, tzinfo=timezone.utc) - timedelta(milliseconds=1)
                
                start_ts = int(m_start.timestamp() * 1000)
                end_ts = int(m_end.timestamp() * 1000)
                
                # Check DB
                # We use a lightweight query
                has_data = False
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT 1 FROM klines WHERE symbol=? AND timestamp BETWEEN ? AND ? LIMIT 1", 
                        (symbol, start_ts, end_ts)
                    )
                    if cursor.fetchone():
                        has_data = True
                
                if not has_data:
                    tasks.append((symbol, year, month))
        
        if not tasks:
            logger.info("All data already present in DB! Skipping download.")
            return

        logger.info(f"Found {len(tasks)} missing monthly chunks. Downloading in parallel...")
        
        # 2. Download in Parallel
        # We use a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Map tasks to futures
            future_to_task = {
                executor.submit(self.vision_loader.download_monthly_data, symbol, year, month, interval): (symbol, year, month)
                for symbol, year, month in tasks
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                symbol, year, month = future_to_task[future]
                completed += 1
                try:
                    df = future.result()
                    if not df.empty:
                        # Save to DB (Main thread writes, safe)
                        self._save_to_db(symbol, df)
                        # Run QC
                        self._run_quality_check(symbol, df)
                        logger.info(f"[{completed}/{len(tasks)}] Loaded {symbol} {year}-{month}")
                    else:
                        logger.warning(f"[{completed}/{len(tasks)}] No data for {symbol} {year}-{month}")
                except Exception as e:
                    logger.error(f"Error processing {symbol} {year}-{month}: {e}")
            
        logger.info("Database seeding complete!")

    def prefetch_all_data(self, symbols: List[str], start: datetime, end: datetime, interval: str = "5m"):
        """
        DEPRECATED: Use seed_from_vision instead.
        Pre-fetch all data for given symbols and date range to avoid repeated API calls.
        Also runs DATA QUALITY CHECK and stores results.
        
        Args:
            symbols: List of symbols to fetch
            start: Start datetime
            end: End datetime  
            interval: Candle interval
        """
        logger.warning("prefetch_all_data is deprecated. Using API to fetch data. Consider using seed_from_vision.")

        logger.info(f"Pre-fetching data for {len(symbols)} symbols from {start} to {end}...")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"[{i}/{len(symbols)}] Fetching {symbol}...")
                # 1. Fetch Data (fills 'klines' table)
                df = self.get_data(symbol, start, end, interval)
                logger.info(f"[{i}/{len(symbols)}] {symbol}: {len(df)} candles cached")
                
                # 2. Run Quality Check (fills 'data_quality' table)
                if not df.empty:
                    self._run_quality_check(symbol, df)
                
                # Add a delay to avoid hitting rate limits (IP ban protection)
                time.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error prefetching {symbol}: {e}")
        
        logger.info(f"Pre-fetch complete! All data cached in {self.db_path}")

    def _ensure_db(self):
        """Initialize SQLite database schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Klines table
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
            
            # Data Quality table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality (
                    symbol TEXT,
                    timestamp INTEGER,
                    is_valid BOOLEAN,
                    quality_issue TEXT,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_qc_symbol_time ON data_quality (symbol, timestamp)")

    def _run_quality_check(self, symbol: str, df: pd.DataFrame):
        """
        Run Data Quality Checker on a DataFrame and store results in DB.
        """
        if df.empty:
            return

        # Run QC
        df_checked = self.qc.check_quality(df, symbol)
        
        # Prepare for DB
        # We want to store: symbol, timestamp, is_valid, quality_issue
        records = []
        for _, row in df_checked.iterrows():
            # Convert timestamp to ms int
            ts = int(row['timestamp'].timestamp() * 1000)
            is_valid = 1 if row['is_valid'] else 0
            issue = row['quality_issue']
            records.append((symbol, ts, is_valid, issue))
            
        if not records:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO data_quality (symbol, timestamp, is_valid, quality_issue)
                VALUES (?, ?, ?, ?)
            """, records)
            logger.info(f"Stored QC results for {symbol}: {len(records)} records")

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
            if self.client is None:
                logger.warning(f"No cached data for {symbol} and no BinanceClient provided (Read-Only Mode)")
                return pd.DataFrame()
                
            logger.info(f"No cached data for {symbol}. Fetching full range from Binance...")
            df_new = self.client.get_historical_klines(symbol, interval, start_time, end_time)
            if not df_new.empty:
                self._save_to_db(symbol, df_new)
            return df_new
        
        # Check edges
        db_min_ts = df_db["timestamp"].min().timestamp() * 1000
        db_max_ts = df_db["timestamp"].max().timestamp() * 1000
        
        # Only fetch if gap is significant (> 1 hour) to avoid spamming API for single candles
        GAP_THRESHOLD_MS = 60 * 60 * 1000 
        
        # Fetch missing head (before DB data)
        if start_ts < db_min_ts and (db_min_ts - start_ts) > GAP_THRESHOLD_MS:
            missing_start = start_time
            missing_end = datetime.fromtimestamp(db_min_ts / 1000, tz=timezone.utc)
            logger.info(f"Fetching missing head for {symbol}: {missing_start} to {missing_end}")
            df_head = self.client.get_historical_klines(symbol, interval, missing_start, missing_end)
            if not df_head.empty:
                self._save_to_db(symbol, df_head)
                
        # Fetch missing tail (after DB data)
        if end_ts > db_max_ts and (end_ts - db_max_ts) > GAP_THRESHOLD_MS:
            missing_start = datetime.fromtimestamp(db_max_ts / 1000, tz=timezone.utc)
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
            # Convert timestamp to datetime and localize to UTC
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize('UTC')
        return df

    def _save_to_db(self, symbol: str, df: pd.DataFrame):
        """Save DataFrame to SQLite."""
        if df.empty:
            return
            
        # Prepare data for insertion
        data = df.copy()
        data["symbol"] = symbol
        # Ensure timestamp is integer milliseconds
        # Convert to int64 first, then to native python int to avoid numpy types in SQLite
        data["timestamp"] = (data["timestamp"].astype("int64") // 10**6).astype(int)
        
        # Convert to list of tuples for sqlite3
        # to_records() can cause issues with numpy types being stored as BLOBs
        records = data[["symbol", "timestamp", "open", "high", "low", "close", "volume"]].values.tolist()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO klines (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, records)
            logger.info(f"Cached {len(records)} records for {symbol}")

    def get_data_with_quality(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Get data with quality flags (is_valid, quality_issue).
        """
        # 1. Get Klines
        df = self.get_data(symbol, start, end)
        if df.empty:
            return df
            
        # 2. Get Quality Flags
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)
        
        query = """
            SELECT timestamp, is_valid, quality_issue
            FROM data_quality
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
        """
        with sqlite3.connect(self.db_path) as conn:
            df_qc = pd.read_sql_query(query, conn, params=(symbol, start_ts, end_ts))
            
        if df_qc.empty:
            # No QC data found? Assume valid.
            df['is_valid'] = True
            df['quality_issue'] = None
            return df
            
        # Convert QC timestamp
        df_qc['timestamp'] = pd.to_datetime(df_qc['timestamp'], unit='ms').dt.tz_localize('UTC')
        
        # Merge
        # Ensure timestamps match exactly
        df = pd.merge(df, df_qc, on='timestamp', how='left')
        
        # Fill NaNs (missing QC data -> assume valid)
        df['is_valid'] = df['is_valid'].fillna(1).astype(bool)
        df['quality_issue'] = df['quality_issue'].fillna('NONE')
        
        return df
