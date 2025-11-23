"""
Binance Vision Data Loader.
Downloads monthly ZIP files from Binance Vision public HTTP endpoints and seeds the database.
URL Format: https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year}-{month}.zip
"""

import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import time
from src.logger import get_logger

logger = get_logger(__name__)

class BinanceVisionLoader:
    """
    Downloads and processes historical data from Binance Vision.
    """
    
    BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
    
    def __init__(self, cache_dir: str = "data/vision_cache"):
        self.cache_dir = cache_dir
        # Ensure cache dir exists? We might process in memory to save space.
        # But caching ZIPs might be useful. For now, in-memory.

    def download_monthly_data(self, symbol: str, year: str, month: str, interval: str = "5m") -> pd.DataFrame:
        """
        Download and parse monthly data for a symbol.
        """
        filename = f"{symbol}-{interval}-{year}-{month}.zip"
        url = f"{self.BASE_URL}/{symbol}/{interval}/{filename}"
        return self._download_url(url, symbol)

    def _download_url(self, url: str, symbol: str) -> pd.DataFrame:
        """Internal method to download a specific URL."""
        # logger.info(f"Downloading {url}...") # Too noisy for parallel
        
        try:
            response = requests.get(url)
            if response.status_code == 404:
                # logger.warning(f"Data not found: {url}")
                return pd.DataFrame()
            if response.status_code != 200:
                logger.error(f"Failed to download {url}: Status {response.status_code}")
                return pd.DataFrame()
                
            # Extract ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Find the CSV file inside
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    return pd.DataFrame()
                csv_filename = csv_files[0]
                
                with z.open(csv_filename) as f:
                    df = pd.read_csv(f)
                    
                    # Rename open_time to timestamp
                    if 'open_time' in df.columns:
                        df.rename(columns={'open_time': 'timestamp'}, inplace=True)
                    
                    # Keep relevant columns
                    needed_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    # Ensure columns exist
                    available_cols = [c for c in needed_cols if c in df.columns]
                    df = df[available_cols]
                    
                    # Convert types
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            df[col] = df[col].astype(float)
                        
                    return df
                    
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return pd.DataFrame()

    def get_months_range(self, start_date: datetime, end_date: datetime) -> List[tuple]:
        """Generate (year, month) tuples for the range."""
        months = []
        current = start_date.replace(day=1)
        # Ensure we cover the end_date's month too
        end_month_first = end_date.replace(day=1)
        
        while current <= end_month_first:
            months.append((str(current.year), f"{current.month:02d}"))
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return months
