import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, replace

from src.backtesting.data_manager import DataManager

from src.binance_client import BinanceClient
from src.formation import FormationManager
from src.copula_model import CopulaModel, SpreadPair
from src.strategy import PairsTradingStrategy
from src.logger import get_logger
from src.backtesting.data_manager import DataManager

logger = get_logger(__name__)

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    pair: str  # "ALT1-ALT2"
    side: str  # "LONG_S1_SHORT_S2" or "SHORT_S1_LONG_S2"
    size_alt1: float
    size_alt2: float
    entry_price_alt1: float
    entry_price_alt2: float
    exit_price_alt1: float = 0.0
    exit_price_alt2: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED

class BacktestEngine:
    def __init__(
        self,
        data_manager: DataManager,
        config: object,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        output_dir: Optional[str] = None
    ):
        self.dm = data_manager
        self.config = config
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.equity = initial_capital
        self.output_dir = output_dir
        
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # State
        self.current_pair: Optional[SpreadPair] = None
        self.strategy: Optional[PairsTradingStrategy] = None
        self.current_trade: Optional[Trade] = None
        self.last_formation_time: Optional[datetime] = None
        
        # Create a mock client that redirects to DataManager
        # This ensures FormationManager uses cached data instead of hitting the API
        class MockBinanceClient:
            def __init__(self, dm):
                self.dm = dm
                
            def get_historical_klines(self, symbol, interval, start_str, end_str):
                # Convert strings/timestamps to datetime if needed
                # FormationManager passes datetime objects usually
                return self.dm.get_data(symbol, start_str, end_str, interval)
                
            def futures_exchange_info(self):
                return self.dm.client.futures_exchange_info()
                
        self.mock_client = MockBinanceClient(self.dm)
        
        # Initialize FormationManager with the mock client
        self.formation_manager = FormationManager(
            binance_client=self.mock_client,
            altcoins=config.trading.altcoins,
            formation_days=config.trading.formation_days,
            volatility_jump_threshold=config.risk_management.volatility_jump_threshold,
            volatility_match_factor=config.risk_management.volatility_match_factor,
        )

    def _get_next_scheduled_formation(self, current_time: datetime) -> datetime:
        """Calculate the next scheduled formation time based on config."""
        target_day_str = self.config.scheduler.formation_day_of_week.lower()
        target_hour = self.config.scheduler.formation_hour
        target_minute = self.config.scheduler.formation_minute
        
        days_map = {
            'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6
        }
        target_day = days_map.get(target_day_str, 0)
        
        # Ensure current_time is UTC-aware
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
            
        # Calculate days ahead
        current_day = current_time.weekday()
        days_ahead = target_day - current_day
        
        # Create candidate time for this week
        # We calculate the date offset first
        candidate_date = current_time.date() + timedelta(days=days_ahead)
        candidate_time = datetime.combine(
            candidate_date, 
            datetime.min.time().replace(hour=target_hour, minute=target_minute)
        ).replace(tzinfo=timezone.utc)
        
        # If candidate is in the past, move to next week
        if candidate_time < current_time:
            candidate_time += timedelta(days=7)
            
        return candidate_time

    def run(self):
        """Run the backtest simulation."""
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # STEP 1: PRE-FETCH ALL DATA (Handled by orchestrator/backtest.py)
        # self.dm.prefetch_all_data(...) -> DEPRECATED
        logger.info("Starting simulation (Data should be pre-seeded)...")
        
        logger.info("All data cached! Starting simulation...")
        
        # STEP 2: RUN SIMULATION
        current_time = self.start_date
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
            
        # Align with schedule
        next_formation_time = self._get_next_scheduled_formation(current_time)
        
        if next_formation_time > current_time:
            logger.info(f"Waiting for first formation scheduled at {next_formation_time}...")
        
        # Pre-fetch all data for BTC (master clock)
        # We iterate based on BTC timestamps to ensure alignment
        btc_data = self.dm.get_data(
            "BTCUSDT", 
            self.start_date - timedelta(days=self.config.trading.formation_days), # Fetch extra for first formation
            self.end_date
        )
        
        # Filter for simulation period
        sim_data = btc_data[btc_data["timestamp"] >= self.start_date]
        timestamps = sim_data["timestamp"].tolist()
        
        logger.info(f"Loaded {len(timestamps)} simulation steps")
        
        for ts in timestamps:
            current_time = ts
            # Ensure current_time is UTC-aware
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            
            # 1. Check for Formation Event
            if current_time >= next_formation_time:
                self._run_formation(current_time)
                self.last_formation_time = current_time
                next_formation_time = current_time + timedelta(days=self.config.trading.trading_days)
                
            # 2. Execute Trading Cycle
            if self.strategy:
                self._process_cycle(current_time)
            
            # 3. Update Equity
            self._update_equity(current_time)

        # Force-close any remaining open positions at end of backtest
        if self.current_trade:
            logger.info(f"Force-closing open position at end of backtest: {self.current_trade.pair}")
            self._close_trade(current_time, reason="END_OF_BACKTEST")
            self._update_equity(current_time)

        logger.info("Backtest complete")


    def _run_formation(self, timestamp: datetime):
        """
        Run formation phase: select best pair.
        """
        logger.info(f"[{timestamp}] Running Formation Phase...")
        
        # Close existing trade if any
        if self.current_trade:
            self._close_trade(timestamp, reason="FORMATION_RESET")
            
        # Define formation window
        start_date = timestamp - timedelta(days=self.config.trading.formation_days)
        end_date = timestamp
        
        # 1. Fetch data for all candidates
        candidates = self.config.trading.altcoins
        valid_candidates = []
        
        # DATA QUALITY CHECK: Filter out pairs with bad data in formation period
        for coin in candidates:
            try:
                df = self.dm.get_data_with_quality(coin, start_date, end_date)
                if df.empty:
                    logger.warning(f"Skipping {coin}: No data for formation")
                    continue
                    
                # Check for invalid rows
                # We assume 'is_valid' column exists from get_data_with_quality
                if 'is_valid' in df.columns:
                    invalid_count = (~df['is_valid']).sum()
                    if invalid_count > 0:
                        logger.warning(f"Skipping {coin}: Found {invalid_count} invalid candles in formation period")
                        continue
                    
                valid_candidates.append(coin)
            except Exception as e:
                logger.error(f"Error checking QC for {coin}: {e}")
                
        if len(valid_candidates) < 2:
            logger.warning("Not enough valid candidates for formation")
            self.current_pair = None
            self.strategy = None
            return

        # Create a temporary config with filtered altcoins
        temp_config = self.config
        from dataclasses import replace
        new_trading = replace(temp_config.trading, altcoins=valid_candidates)
        temp_config = replace(temp_config, trading=new_trading)
        
        # Initialize Formation with filtered config
        # Note: We use self.mock_client if available, or self.dm.client
        # But FormationManager takes binance_client.
        # We should use the same client as __init__ used.
        formation = FormationManager(
            binance_client=self.mock_client,
            altcoins=valid_candidates,
            formation_days=temp_config.trading.formation_days,
            volatility_jump_threshold=temp_config.risk_management.volatility_jump_threshold,
            volatility_match_factor=temp_config.risk_management.volatility_match_factor,
        )
        
        try:
            spread_pair = formation.run_formation(end_time=timestamp)
            
            if spread_pair:
                self.current_pair = spread_pair
                self.strategy = PairsTradingStrategy(
                    spread_pair=spread_pair,
                    entry_threshold=self.config.trading.entry_threshold,
                    exit_threshold=self.config.trading.exit_threshold,
                    capital_per_leg=self.config.trading.capital_per_leg
                )
                logger.info(f"Selected pair: {spread_pair.alt1}-{spread_pair.alt2} with |τ|={spread_pair.tau:.4f}")
            else:
                logger.info("No suitable pair found.")
                self.current_pair = None
                self.strategy = None
                
        except Exception as e:
            logger.error(f"Formation failed: {e}")
            self.current_pair = None
            self.strategy = None

    def _process_cycle(self, current_time: datetime):
        """Process a single trading cycle."""
        # NO-TRADE ZONE: Do not trade for 15 minutes after formation
        if self.last_formation_time:
            time_since_formation = (current_time - self.last_formation_time).total_seconds() / 60
            if time_since_formation < 15:
                return

        # Get current prices
        prices = self._get_current_prices(current_time)
        if not prices:
            return

        # RISK MANAGEMENT: Check stop-loss and time-based exit for open trades
        if self.current_trade:
            # 1. Stop-loss check (percentage-based)
            unrealized_pnl = self._calculate_unrealized_pnl(current_time, prices)
            position_value = self.config.trading.capital_per_leg * 2  # Both legs
            stop_loss_threshold = -position_value * self.config.risk_management.stop_loss_pct
            
            if unrealized_pnl < stop_loss_threshold:
                logger.warning(
                    f"[{current_time}] STOP-LOSS triggered: PnL=${unrealized_pnl:.2f} "
                    f"< ${stop_loss_threshold:.2f} "
                    f"({self.config.risk_management.stop_loss_pct:.1%} of ${position_value:.2f})"
                )
                self._close_trade(current_time, "STOP_LOSS")
                return
            
            # 2. Time-based exit check
            trade_duration_hours = (current_time - self.current_trade.entry_time).total_seconds() / 3600
            if trade_duration_hours > self.config.risk_management.max_trade_duration_hours:
                logger.warning(
                    f"[{current_time}] TIME-BASED EXIT: Trade duration {trade_duration_hours:.1f}h "
                    f"> {self.config.risk_management.max_trade_duration_hours}h"
                )
                self._close_trade(current_time, "TIME_EXIT")
                return

        # Generate signal
        signal_obj = self.strategy.generate_signal(
            prices['BTCUSDT'],
            prices[self.current_pair.alt1],
            prices[self.current_pair.alt2]
        )
        
        signal = signal_obj.signal
        
        # Execute signal
        if signal in ['LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2']:
            if not self.current_trade:
                self._open_trade(current_time, signal, prices)
        elif signal == 'CLOSE':
            if self.current_trade:
                self._close_trade(current_time, "SIGNAL_EXIT")

    def _update_equity(self, current_time: datetime):
        """Track equity curve with unrealized PnL."""
        unrealized_pnl = 0
        
        if self.current_trade:
            prices = self._get_current_prices(current_time)
            if prices:
                # Calculate mark-to-market PnL
                temp_trade = Trade(
                    entry_time=self.current_trade.entry_time,
                    exit_time=current_time,
                    pair=self.current_trade.pair,
                    side=self.current_trade.side,
                    size_alt1=self.current_trade.size_alt1,
                    size_alt2=self.current_trade.size_alt2,
                    entry_price_alt1=self.current_trade.entry_price_alt1,
                    entry_price_alt2=self.current_trade.entry_price_alt2,
                    exit_price_alt1=prices[self.current_pair.alt1],
                    exit_price_alt2=prices[self.current_pair.alt2]
                )
                unrealized_pnl = self._calculate_pnl(temp_trade)
        
        self.equity_curve.append({
            "timestamp": current_time,
            "equity": self.equity + unrealized_pnl,
            "cash": self.equity,
            "unrealized_pnl": unrealized_pnl
        })

    def _get_current_prices(self, timestamp: datetime) -> Dict[str, float]:
        """
        Get prices for all symbols at a specific timestamp.
        STRICT MODE: 
        1. Target the candle that JUST COMPLETED (Open Time = timestamp - 5m)
        2. Require EXACT timestamp match (no stale data)
        3. No lookahead (do not use candle opening at timestamp)
        4. QUALITY CHECK: Reject invalid candles (static/gaps)
        """
        symbols = ['BTCUSDT', self.current_pair.alt1, self.current_pair.alt2]
        prices = {}
        
        # We want the candle that closed at 'timestamp'.
        # Since timestamps are Open Times, we want the candle with Open Time = timestamp - 5m
        target_open_time = timestamp - timedelta(minutes=5)
        
        for symbol in symbols:
            try:
                # Fetch exactly the candle we need WITH QUALITY FLAGS
                # Use get_data_with_quality from DM
                df = self.dm.get_data_with_quality(
                    symbol,
                    target_open_time,
                    target_open_time
                )
                
                if df.empty:
                    # Try fetching from API if missing in DB (handled by DM, but double check)
                    # If still empty, it's a gap
                    logger.warning(f"No data for {symbol} at {timestamp} (Target Open: {target_open_time})")
                    return {}
                
                # STRICT: Must have exact match
                candle = df[df['timestamp'] == target_open_time]
                if candle.empty:
                    logger.warning(f"Missing exact candle for {symbol} at {target_open_time}")
                    return {}
                
                latest_candle = candle.iloc[0]
                
                # QUALITY CHECK
                if not latest_candle.get('is_valid', True): # Default True if column missing
                    issue = latest_candle.get('quality_issue', 'UNKNOWN')
                    logger.warning(f"Invalid data for {symbol} at {timestamp}: {issue}")
                    return {} # Treat as missing data
                
                # STRICT: Price must be valid
                price = float(latest_candle['close'])
                if pd.isna(price) or price <= 0:
                    raise ValueError(f"Invalid price for {symbol}: {price}")
                
                prices[symbol] = price
                
            except Exception as e:
                logger.error(f"Price fetch failed for {symbol} at {timestamp}: {e}")
                return {}
        
        return prices
    
    
    def _open_trade(self, timestamp: datetime, signal: str, prices: Dict[str, float]):
        """
        Open a new trade.
        STRICT MODE: Validates prices before opening.
        """
        # STRICT: Validate all prices are positive
        for symbol, price in prices.items():
            if price <= 0 or pd.isna(price):
                raise ValueError(f"Invalid price for {symbol}: {price}")
        
        pair_name = f"{self.current_pair.alt1}-{self.current_pair.alt2}"
        
        # Calculate position sizes using strategy
        target_positions = self.strategy.get_target_positions(signal, prices)
        
        # Extract sizes (quantity)
        alt1_size = target_positions[self.current_pair.alt1][1]
        alt2_size = target_positions[self.current_pair.alt2][1]
        
        # STRICT: Validate quantities are positive
        if alt1_size <= 0 or alt2_size <= 0:
            raise ValueError(f"Invalid position sizes: alt1={alt1_size}, alt2={alt2_size}")
        
        self.current_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            pair=pair_name,
            side=signal,
            size_alt1=alt1_size,
            size_alt2=alt2_size,
            entry_price_alt1=prices[self.current_pair.alt1],
            entry_price_alt2=prices[self.current_pair.alt2],
        )
        
        logger.debug(
            f"[{timestamp}] OPEN {signal}: "
            f"{alt1_size:.2f} {self.current_pair.alt1} @ {prices[self.current_pair.alt1]}, "
            f"{alt2_size:.2f} {self.current_pair.alt2} @ {prices[self.current_pair.alt2]}"
        )

    def _close_trade(self, timestamp: datetime, reason: str):
        """Close the current trade."""
        if not self.current_trade:
            return
        
        # Get exit prices
        prices = self._get_current_prices(timestamp)
        if not prices:
            logger.error(f"Cannot close trade at {timestamp}: no price data")
            return
        
        # Update trade with exit info
        self.current_trade.exit_time = timestamp
        self.current_trade.exit_price_alt1 = prices[self.current_pair.alt1]
        self.current_trade.exit_price_alt2 = prices[self.current_pair.alt2]
        
        # Calculate PnL
        pnl = self._calculate_pnl(self.current_trade)
        
        self.equity += pnl
        self.trades.append(self.current_trade)
        
        logger.debug(f"[{timestamp}] CLOSE ({reason}) PnL=${pnl:.2f}")
        
        self.current_trade = None

    def _calculate_pnl(self, trade: Trade) -> float:
        """Calculate trade PnL using centralized strategy logic."""
        if not self.strategy:
            # Fallback if strategy not initialized (shouldn't happen during active trade)
            return 0.0
            
        entry_prices = {
            self.current_pair.alt1: trade.entry_price_alt1,
            self.current_pair.alt2: trade.entry_price_alt2
        }
        
        exit_prices = {
            self.current_pair.alt1: trade.exit_price_alt1,
            self.current_pair.alt2: trade.exit_price_alt2
        }
        
        quantities = {
            self.current_pair.alt1: trade.size_alt1,
            self.current_pair.alt2: trade.size_alt2
        }
        
        return self.strategy.calculate_pnl(entry_prices, exit_prices, quantities, trade.side)

    def _calculate_unrealized_pnl(self, current_time: datetime, prices: Dict[str, float]) -> float:
        """Calculate unrealized PnL for the current open trade."""
        if not self.current_trade:
            return 0.0
        
        # Create a temporary trade object with current prices as exit prices
        temp_trade = Trade(
            entry_time=self.current_trade.entry_time,
            exit_time=current_time,
            pair=self.current_trade.pair,
            side=self.current_trade.side,
            size_alt1=self.current_trade.size_alt1,
            size_alt2=self.current_trade.size_alt2,
            entry_price_alt1=self.current_trade.entry_price_alt1,
            entry_price_alt2=self.current_trade.entry_price_alt2,
            exit_price_alt1=prices[self.current_pair.alt1],
            exit_price_alt2=prices[self.current_pair.alt2],
        )
        
        return self._calculate_pnl(temp_trade)


    def run_parallel(self):
        """Run the backtest simulation in parallel using multiprocessing."""
        logger.info(f"Starting PARALLEL backtest from {self.start_date} to {self.end_date}")
        
        # STEP 1: PRE-FETCH ALL DATA (Handled by orchestrator/backtest.py)
        logger.info("Starting parallel simulation (Data should be pre-seeded)...")
        logger.info("All data cached! Starting parallel simulation...")

        # STEP 2: PREPARE CHUNKS
        # We split time into chunks based on formation schedule
        chunks = []
        current_chunk_start = self._get_next_scheduled_formation(self.start_date)
        
        if current_chunk_start > self.start_date:
             logger.info(f"Skipping initial period {self.start_date} -> {current_chunk_start} to align with schedule")
        
        while current_chunk_start < self.end_date:
            current_chunk_end = min(current_chunk_start + timedelta(days=self.config.trading.trading_days), self.end_date)
            chunks.append((current_chunk_start, current_chunk_end))
            current_chunk_start = current_chunk_end
            
        logger.info(f"Split backtest into {len(chunks)} chunks")

        # STEP 3: EXECUTE IN PARALLEL
        # We need to pass configuration and DB path to workers
        # Note: We can't pass the DataManager instance itself as it contains a sqlite connection
        db_path = self.dm.db_path
        api_key = self.config.binance.api_key
        api_secret = self.config.binance.api_secret
        
        all_trades = []
        
        # Use fewer workers than CPU count to avoid DB contention if any
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    process_backtest_chunk,
                    start,
                    end,
                    self.config,
                    db_path,
                    api_key,
                    api_secret,
                    is_final_chunk=(end == self.end_date),  # Only force-close on final chunk
                    output_dir=self.output_dir
                ): (start, end) for start, end in chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk_start, chunk_end = future_to_chunk[future]
                try:
                    chunk_trades, chunk_pnl_series = future.result()
                    all_trades.extend(chunk_trades)
                    logger.info(f"Chunk {chunk_start} -> {chunk_end} completed: {len(chunk_trades)} trades")
                except Exception as e:
                    logger.error(f"Chunk {chunk_start} -> {chunk_end} failed: {e}", exc_info=True)

        # STEP 4: RECONSTRUCT EQUITY CURVE
        # Sort trades by entry time
        self.trades = sorted(all_trades, key=lambda t: t.entry_time)
        
        # Reconstruct equity curve sequentially
        # This is fast enough since we just iterate trades/time, not processing signals
        self.equity = self.capital
        self.equity_curve = []
        
        # We can just use the trades to build the equity curve points
        # Or we can use the detailed PnL series returned by chunks if we want high res
        # For now, let's just rebuild a simple equity curve based on realized PnL
        # To get a nice curve, we'd need the chunk_pnl_series.
        # Let's assume we want high resolution equity curve.
        
        # But wait, stitching PnL series is tricky because equity compounds (if we reinvest)
        # or adds up (if fixed capital).
        # Current logic uses fixed capital per leg, so PnL is additive.
        
        # Let's just set the final trades list. The standard reporting tools
        # usually reconstruct the curve from the trade list anyway.
        # But for our internal self.equity_curve, we can just append the trade results.
        
        current_equity = self.capital
        for trade in self.trades:
            current_equity += trade.pnl
            self.equity_curve.append({
                "timestamp": trade.exit_time,
                "equity": current_equity,
                "cash": current_equity, # Simplified
                "unrealized_pnl": 0
            })
            
        self.equity = current_equity
        logger.info(f"Parallel backtest complete. Final Equity: ${self.equity:.2f}")


def process_backtest_chunk(
    start_time: datetime,
    end_time: datetime,
    config: object,
    db_path: str,
    api_key: str,
    api_secret: str,
    is_final_chunk: bool = False,
    output_dir: Optional[str] = None
) -> Tuple[List[Trade], List[Dict]]:
    """
    Process a single backtest chunk (formation period) in a separate process.
    """
    # Re-initialize dependencies in the worker process
    # 1. DataManager (READ-ONLY to avoid DB locking issues)
    # We need a BinanceClient for DataManager, even if we only read from DB
    # We can use a mock or real one.
    client = BinanceClient(api_key, api_secret)
    dm = DataManager(db_path, client, read_only=True)
    
    # Mock client for FormationManager to use DM
    class MockBinanceClient:
        def __init__(self, dm):
            self.dm = dm
        def get_historical_klines(self, symbol, interval, start_str, end_str):
            return self.dm.get_data(symbol, start_str, end_str, interval)
        def futures_exchange_info(self):
            return {} # Not needed for formation
            
    mock_client = MockBinanceClient(dm)
    
    # 2. FormationManager
    fm = FormationManager(
        binance_client=mock_client,
        altcoins=config.trading.altcoins,
        formation_days=config.trading.formation_days,
        volatility_jump_threshold=config.risk_management.volatility_jump_threshold,
        volatility_match_factor=config.risk_management.volatility_match_factor,
    )
    
    # 3. Run Formation
    # Formation uses data BEFORE start_time
    spread_pair = fm.run_formation(end_time=start_time, output_dir=output_dir)
    
    if not spread_pair:
        return [], []
        
    # 4. Initialize Strategy
    strategy = PairsTradingStrategy(
        spread_pair=spread_pair,
        entry_threshold=config.trading.entry_threshold,
        exit_threshold=config.trading.exit_threshold,
        capital_per_leg=config.trading.capital_per_leg
    )
    # We still need the model for vectorized signal generation
    model = strategy.model
    
    # 5. Fetch Data for the Chunk
    # We need BTC, Alt1, Alt2 for the whole chunk
    # Fetching all at once is much faster
    btc_df = dm.get_data("BTCUSDT", start_time, end_time)
    alt1_df = dm.get_data(spread_pair.alt1, start_time, end_time)
    alt2_df = dm.get_data(spread_pair.alt2, start_time, end_time)
    
    # Align data
    # Inner join on timestamp
    df = pd.merge(btc_df[['timestamp', 'close']], alt1_df[['timestamp', 'close']], on='timestamp', suffixes=('_btc', '_alt1'))
    df = pd.merge(df, alt2_df[['timestamp', 'close']], on='timestamp')
    # After first merge: timestamp, close_btc, close_alt1
    # After second merge: timestamp, close_btc, close_alt1, close
    df = df.rename(columns={'close_btc': 'btc', 'close_alt1': 'alt1', 'close': 'alt2'})
    df = df.sort_values('timestamp')
    
    if df.empty:
        return [], []
        
    timestamps = df['timestamp'].values
    btc_prices = df['btc'].values
    alt1_prices = df['alt1'].values
    alt2_prices = df['alt2'].values
    
    # 6. Vectorized Signal Generation
    signals_df = model.generate_signals_vectorized(btc_prices, alt1_prices, alt2_prices)
    signals = signals_df['signal'].values
    
    # 7. Simulate Trades
    trades = []
    current_trade = None
    
    # Iterate through aligned data
    # We still need to iterate to handle trade state, but we don't calculate signals
    for i in range(len(timestamps)):
        ts = pd.to_datetime(timestamps[i]).tz_localize('UTC') if pd.to_datetime(timestamps[i]).tz is None else pd.to_datetime(timestamps[i])
        
        # NO-TRADE ZONE: Do not trade for 15 minutes after formation (start_time)
        # start_time is the formation time for this chunk
        if (ts - start_time).total_seconds() < 15 * 60:
            continue
            
        signal = signals[i]
        
        # Current prices
        prices = {
            "BTCUSDT": btc_prices[i],
            spread_pair.alt1: alt1_prices[i],
            spread_pair.alt2: alt2_prices[i]
        }
        
        # Risk Management & Exit
        if current_trade:
            # Check Stop Loss
            # We need to calculate PnL. 
            # Re-implementing simplified PnL calc here to avoid dependency on Engine methods
            
            # Calculate Unrealized PnL
            entry_prices = {
                spread_pair.alt1: current_trade.entry_price_alt1,
                spread_pair.alt2: current_trade.entry_price_alt2
            }
            current_prices = {
                spread_pair.alt1: prices[spread_pair.alt1],
                spread_pair.alt2: prices[spread_pair.alt2]
            }
            quantities = {
                spread_pair.alt1: current_trade.size_alt1,
                spread_pair.alt2: current_trade.size_alt2
            }
            
            unrealized_pnl = strategy.calculate_pnl(entry_prices, current_prices, quantities, current_trade.side)
            
            position_value = config.trading.capital_per_leg * 2
            stop_loss_threshold = -position_value * config.risk_management.stop_loss_pct
            
            # Check Stop Loss
            if unrealized_pnl < stop_loss_threshold:
                current_trade.exit_time = ts
                current_trade.exit_price_alt1 = prices[spread_pair.alt1]
                current_trade.exit_price_alt2 = prices[spread_pair.alt2]
                current_trade.pnl = unrealized_pnl
                current_trade.status = "CLOSED"
                trades.append(current_trade)
                current_trade = None
                continue
                
            # Check Time Exit
            duration_hours = (ts - current_trade.entry_time).total_seconds() / 3600
            if duration_hours > config.risk_management.max_trade_duration_hours:
                current_trade.exit_time = ts
                current_trade.exit_price_alt1 = prices[spread_pair.alt1]
                current_trade.exit_price_alt2 = prices[spread_pair.alt2]
                current_trade.pnl = unrealized_pnl
                current_trade.status = "CLOSED"
                trades.append(current_trade)
                current_trade = None
                continue
                
            # Check Signal Exit
            if signal == "CLOSE":
                current_trade.exit_time = ts
                current_trade.exit_price_alt1 = prices[spread_pair.alt1]
                current_trade.exit_price_alt2 = prices[spread_pair.alt2]
                current_trade.pnl = unrealized_pnl
                current_trade.status = "CLOSED"
                trades.append(current_trade)
                current_trade = None
                continue
                
        # Entry
        elif signal in ['LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2']:
            # Open new trade
            pair_name = f"{spread_pair.alt1}-{spread_pair.alt2}"
            alt1_size = config.trading.capital_per_leg / prices[spread_pair.alt1]
            alt2_size = config.trading.capital_per_leg / prices[spread_pair.alt2]
            
            current_trade = Trade(
                entry_time=ts,
                exit_time=None,
                pair=pair_name,
                side=signal,
                size_alt1=alt1_size,
                size_alt2=alt2_size,
                entry_price_alt1=prices[spread_pair.alt1],
                entry_price_alt2=prices[spread_pair.alt2]
            )
            
    # Force close at end of chunk (formation boundary)
    # Each chunk ends at a formation time, so any open trade must be closed
    if current_trade:
        # We close at the last available price
        current_trade.exit_time = pd.to_datetime(timestamps[-1]).tz_localize('UTC') if pd.to_datetime(timestamps[-1]).tz is None else pd.to_datetime(timestamps[-1])
        current_trade.exit_price_alt1 = alt1_prices[-1]
        current_trade.exit_price_alt2 = alt2_prices[-1]
        
        # Recalculate PnL
        entry_prices = {
            spread_pair.alt1: current_trade.entry_price_alt1,
            spread_pair.alt2: current_trade.entry_price_alt2
        }
        exit_prices = {
            spread_pair.alt1: current_trade.exit_price_alt1,
            spread_pair.alt2: current_trade.exit_price_alt2
        }
        quantities = {
            spread_pair.alt1: current_trade.size_alt1,
            spread_pair.alt2: current_trade.size_alt2
        }
        
        current_trade.pnl = strategy.calculate_pnl(entry_prices, exit_prices, quantities, current_trade.side)
        current_trade.status = "CLOSED"
        trades.append(current_trade)
        
    return trades, [] # We skip detailed PnL series for now to save bandwidth

