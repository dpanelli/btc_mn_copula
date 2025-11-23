"""Binance Futures API client wrapper."""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from src.logger import get_logger

logger = get_logger(__name__)


class BinanceClient:
    """Wrapper for Binance Futures API operations."""

    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """
        Initialize Binance client.

        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
            testnet: Use testnet if True, live trading if False
        """
        self.testnet = testnet
        if testnet:
            self.client = Client(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True,
            )
            logger.info("Initialized Binance Futures TESTNET client")
        else:
            self.client = Client(api_key=api_key, api_secret=api_secret)
            logger.info("Initialized Binance Futures LIVE client (Public Data Mode)")

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical kline/candlestick data with automatic pagination, throttling, and retry logic.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '5m', '1h')
            start_time: Start time for historical data
            end_time: End time for historical data (default: now)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            if end_time is None:
                end_time = datetime.now(timezone.utc)

            logger.debug(
                f"Fetching klines for {symbol} from {start_time} to {end_time}, interval={interval}"
            )

            # Convert end time to milliseconds
            end_ms = int(end_time.timestamp() * 1000)

            # Collect all klines across multiple requests (pagination)
            all_klines = []
            current_start = start_time
            batch_count = 0

            while current_start < end_time:
                batch_count += 1
                start_ms = int(current_start.timestamp() * 1000)

                try:
                    # Fetch batch of up to 1500 klines
                    klines = self.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=start_ms,
                        endTime=end_ms,
                        limit=1500,  # Max per request
                    )

                    if not klines:
                        # No more data available
                        break

                    all_klines.extend(klines)
                    logger.debug(
                        f"Batch {batch_count}: Fetched {len(klines)} klines for {symbol}"
                    )

                    # Check if we got less than 1500 klines (reached the end)
                    if len(klines) < 1500:
                        break

                    # Update start time for next batch: use last candle's close_time + 1ms
                    last_close_time_ms = int(klines[-1][6])
                    current_start = datetime.fromtimestamp(
                        last_close_time_ms / 1000, tz=timezone.utc
                    ) + timedelta(milliseconds=1)

                    # Safety check: prevent infinite loop
                    if current_start >= end_time:
                        break
                    
                    # THROTTLING: Sleep 1.0s between requests
                    time.sleep(1.0)

                except BinanceAPIException as e:
                    if e.code == -1003: # Too many requests
                        logger.warning(f"Rate limit hit for {symbol}. Sleeping 60s...")
                        time.sleep(60)
                        continue
                    else:
                        raise e
                except Exception as e:
                    logger.error(f"Error fetching batch for {symbol}: {e}")
                    # Retry once?
                    time.sleep(5)
                    continue

            if not all_klines:
                logger.warning(f"No klines returned for {symbol}")
                return pd.DataFrame()

            logger.info(
                f"Fetched {len(all_klines)} total klines for {symbol} across {batch_count} batch(es)"
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                all_klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            # Filter out incomplete candles (last candle might still be forming)
            # A candle is complete when close_time <= current time
            if len(df) > 0:
                current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                df["close_time"] = pd.to_numeric(df["close_time"], errors="coerce")

                # Keep only candles that have closed
                initial_count = len(df)
                df = df[df["close_time"] <= current_time_ms].copy()

                if len(df) < initial_count:
                    logger.debug(
                        f"Filtered out {initial_count - len(df)} incomplete candle(s) for {symbol}"
                    )

            # Keep only relevant columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize('UTC')
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # STRICT VALIDATION: Check for NaN or invalid values
            if df[["open", "high", "low", "close", "volume"]].isna().any().any():
                nan_count = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
                logger.error(f"Found {nan_count} NaN values in {symbol} data - REJECTING")
                raise ValueError(f"Invalid data for {symbol}: contains NaN values")
            
            # STRICT VALIDATION: Check for zero or negative prices
            price_cols = ["open", "high", "low", "close"]
            if (df[price_cols] <= 0).any().any():
                logger.error(f"Found zero or negative prices in {symbol} data - REJECTING")
                raise ValueError(f"Invalid data for {symbol}: contains zero/negative prices")
            
            # STRICT VALIDATION: Check for gaps in timestamps (should be 5 minutes apart)
            if len(df) > 1:
                time_diffs = df['timestamp'].diff().dt.total_seconds() / 60
                expected_diff = 5  # 5-minute candles
                gaps = time_diffs[(time_diffs > expected_diff * 1.5) & (time_diffs.notna())]
                if len(gaps) > 0:
                    logger.warning(
                        f"Found {len(gaps)} gaps in {symbol} data (expected 5min intervals). "
                        f"Max gap: {gaps.max():.0f} minutes"
                    )

            logger.info(f"Returning {len(df)} complete klines for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            raise

    def get_current_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price as float
        """
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            raise

    def futures_exchange_info(self) -> Dict:
        """
        Get exchange information.

        Returns:
            Dictionary with exchange info
        """
        try:
            return self.client.futures_exchange_info()
        except BinanceAPIException as e:
            logger.error(f"Error fetching exchange info: {e}")
            raise

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair symbol
            leverage: Leverage multiplier (1-125)
        """
        try:
            result = self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"Set leverage for {symbol} to {leverage}x: {result}")
        except BinanceAPIException as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            raise

    def place_market_order(
        self, symbol: str, side: str, quantity: float, reduce_only: bool = False
    ) -> Dict:
        """
        Place a market order.

        Args:
            symbol: Trading pair symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Order quantity in base asset
            reduce_only: If True, order can only reduce position (prevents flipping), default: False

        Returns:
            Order response dict
        """
        try:
            from binance.enums import ORDER_TYPE_MARKET
            
            logger.info(
                f"Placing market {side} order for {symbol}: {quantity} "
                f"(reduceOnly={reduce_only})"
            )

            order_params = {
                "symbol": symbol,
                "side": side,
                "type": ORDER_TYPE_MARKET,
                "quantity": quantity,
            }

            # Add reduceOnly if requested (string value required by Binance API)
            if reduce_only:
                order_params["reduceOnly"] = "true"

            order = self.client.futures_create_order(**order_params)

            logger.info(f"Order placed successfully: {order}")
            return order

        except BinanceAPIException as e:
            logger.error(f"Error placing market order for {symbol}: {e}")
            raise

    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Position dict with keys: symbol, positionAmt, entryPrice, unRealizedProfit
            Returns None if no position exists
        """
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for pos in positions:
                if float(pos["positionAmt"]) != 0:
                    logger.debug(f"Position for {symbol}: {pos}")
                    return {
                        "symbol": pos["symbol"],
                        "position_amt": float(pos["positionAmt"]),
                        "entry_price": float(pos["entryPrice"]),
                        "unrealized_pnl": float(pos["unRealizedProfit"]),
                        "leverage": int(pos.get("leverage", 1)),  # Default to 1x if not available
                    }
            logger.debug(f"No position for {symbol}")
            return None

        except BinanceAPIException as e:
            logger.error(f"Error fetching position for {symbol}: {e}")
            raise

    def close_position(self, symbol: str) -> Optional[Dict]:
        """
        Close an open position by placing an opposing market order with reduceOnly=True.

        Uses reduceOnly flag to ensure the order can ONLY reduce the position,
        preventing accidental position flips due to timing, rounding, or state issues.

        Args:
            symbol: Trading pair symbol

        Returns:
            Order response dict, or None if no position to close
        """
        try:
            from binance.enums import SIDE_SELL, SIDE_BUY
            
            position = self.get_position(symbol)
            if position is None:
                logger.info(f"No position to close for {symbol}")
                return None

            position_amt = position["position_amt"]
            if position_amt == 0:
                logger.info(f"Position already flat for {symbol}")
                return None

            # Determine side to close position (opposite of current)
            side = SIDE_SELL if position_amt > 0 else SIDE_BUY
            quantity = abs(position_amt)

            logger.info(f"Closing position for {symbol}: {side} {quantity} (reduceOnly=True)")
            return self.place_market_order(symbol, side, quantity, reduce_only=True)

        except BinanceAPIException as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            raise

    def get_account_balance(self) -> float:
        """
        Get USDT balance from Futures account.

        Returns:
            Available USDT balance
        """
        try:
            account = self.client.futures_account()
            for asset in account["assets"]:
                if asset["asset"] == "USDT":
                    balance = float(asset["availableBalance"])
                    logger.debug(f"USDT balance: {balance}")
                    return balance
            return 0.0
        except BinanceAPIException as e:
            logger.error(f"Error fetching account balance: {e}")
            raise

    def calculate_position_size(
        self, symbol: str, capital_usdt: float, leverage: int
    ) -> float:
        """
        Calculate position size in base currency given capital and leverage.

        Args:
            symbol: Trading pair symbol
            capital_usdt: Capital to allocate in USDT
            leverage: Leverage multiplier

        Returns:
            Position size in base currency (rounded to symbol precision)
        """
        try:
            price = self.get_current_price(symbol)
            # Position value = capital * leverage
            position_value = capital_usdt * leverage
            # Position size in base currency
            position_size = position_value / price

            # Get symbol info for precision
            exchange_info = self.client.futures_exchange_info()
            for s in exchange_info["symbols"]:
                if s["symbol"] == symbol:
                    # Get quantity precision
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step_size = float(f["stepSize"])
                            # Round to step size
                            position_size = round(position_size / step_size) * step_size
                            break
                    break

            logger.debug(
                f"Position size for {symbol}: {position_size} "
                f"(capital={capital_usdt}, leverage={leverage}, price={price})"
            )
            return position_size

        except BinanceAPIException as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            raise
