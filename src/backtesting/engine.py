"""
Backtest Engine.
Simulates the strategy over historical data with rolling formation periods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.formation import FormationManager
from src.copula_model import CopulaModel, SpreadPair
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
        initial_capital: float = 10000.0
    ):
        self.dm = data_manager
        self.config = config
        self.start_date = start_date
        self.end_date = end_date
        self.capital = initial_capital
        self.equity = initial_capital
        
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        
        # State
        self.current_pair: Optional[SpreadPair] = None
        self.copula_model: Optional[CopulaModel] = None
        self.current_trade: Optional[Trade] = None
        
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

    def run(self):
        """Run the backtest simulation."""
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # STEP 1: PRE-FETCH ALL DATA (avoiding thousands of small API calls)
        all_symbols = ['BTCUSDT'] + self.config.trading.altcoins
        logger.info(f"Pre-fetching data for {len(all_symbols)} symbols...")
        
        self.dm.prefetch_all_data(
            symbols=all_symbols,
            start=self.start_date - timedelta(days=self.config.trading.formation_days),
            end=self.end_date,
            interval="5m"
        )
        
        logger.info("All data cached! Starting simulation...")
        
        # STEP 2: RUN SIMULATION
        current_time = self.start_date
        next_formation_time = current_time
        
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
            
            # 1. Check for Formation Event
            if current_time >= next_formation_time:
                self._run_formation(current_time)
                next_formation_time = current_time + timedelta(days=7) # Weekly formation
                
            # 2. Execute Trading Cycle
            if self.copula_model:
                self._process_cycle(current_time)
            
            # 3. Update Equity
            self._update_equity(current_time)

        logger.info("Backtest complete")


    def _run_formation(self, current_time: datetime):
        """Run formation phase at current_time."""
        logger.info(f"[{current_time}] Running Formation Phase...")
        
        # Close existing trade if any
        if self.current_trade:
            self._close_trade(current_time, reason="FORMATION_RESET")
            
        # Run formation with historical end time
        try:
            spread_pair = self.formation_manager.run_formation(end_time=current_time)
            
            if spread_pair:
                self.current_pair = spread_pair
                self.copula_model = CopulaModel(
                    spread_pair,
                    entry_threshold=self.config.trading.entry_threshold,
                    exit_threshold=self.config.trading.exit_threshold
                )
                logger.info(
                    f"Selected pair: {spread_pair.alt1} - {spread_pair.alt2} "
                    f"(tau={spread_pair.tau:.4f}, rho={spread_pair.rho:.4f})"
                )
            else:
                logger.warning("Formation failed - no suitable pairs found")
                self.current_pair = None
                self.copula_model = None
        except Exception as e:
            logger.error(f"Formation error: {e}")
            self.current_pair = None
            self.copula_model = None

    def _process_cycle(self, current_time: datetime):
        """Process a single trading cycle."""
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
        signal_data = self.copula_model.generate_signal(
            prices['BTCUSDT'],
            prices[self.current_pair.alt1],
            prices[self.current_pair.alt2]
        )
        
        signal = signal_data['signal']
        
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
        """Get prices for all active assets at timestamp."""
        if not self.current_pair:
            return {}
        
        symbols = ['BTCUSDT', self.current_pair.alt1, self.current_pair.alt2]
        prices = {}
        
        for symbol in symbols:
            try:
                # Request exact timestamp only to avoid triggering "missing tail" fetches
                # The DataManager handles single-point lookups efficiently from cache
                df = self.dm.get_data(symbol, timestamp, timestamp)
                
                if df.empty:
                    # If exact match fails, try a small window (e.g. 1 minute back)
                    # This handles potential slight timestamp misalignments
                    df = self.dm.get_data(symbol, timestamp - timedelta(minutes=1), timestamp)
                
                if df.empty:
                    logger.warning(f"No data for {symbol} at {timestamp}")
                    return {}
                
                # Get close price at timestamp (or closest previous)
                # Sort by timestamp desc to get latest
                df = df.sort_values('timestamp', ascending=False)
                prices[symbol] = float(df.iloc[0]['close'])
            except Exception as e:
                logger.error(f"Error getting price for {symbol} at {timestamp}: {e}")
                return {}
        
        return prices
    
    
    def _open_trade(self, timestamp: datetime, signal: str, prices: Dict[str, float]):
        """Open a new trade."""
        pair_name = f"{self.current_pair.alt1}-{self.current_pair.alt2}"
        
        # Calculate position sizes
        alt1_size = self.config.trading.capital_per_leg / prices[self.current_pair.alt1]
        alt2_size = self.config.trading.capital_per_leg / prices[self.current_pair.alt2]
        
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
        
        logger.info(
            f"[{timestamp}] OPEN {signal}: "
            f"{alt1_size:.2f} {self.current_pair.alt1} @ {prices[self.current_pair.alt1]}, "
            f"{alt2_size:.2f} {self.current_pair.alt2} @ {prices[self.current_pair.alt2]}"
        )

    def _close_trade(self, timestamp: datetime, reason: str):
        """Close current trade and calculate PnL."""
        if not self.current_trade:
            return
        
        # Get exit prices
        prices = self._get_current_prices(timestamp)
        if not prices:
            logger.warning(f"Cannot close trade - no prices at {timestamp}")
            return
        
        self.current_trade.exit_time = timestamp
        self.current_trade.exit_price_alt1 = prices[self.current_pair.alt1]
        self.current_trade.exit_price_alt2 = prices[self.current_pair.alt2]
        
        # Calculate PnL
        pnl = self._calculate_pnl(self.current_trade)
        self.current_trade.pnl = pnl
        self.current_trade.status = "CLOSED"
        
        self.equity += pnl
        self.trades.append(self.current_trade)
        
        logger.info(f"[{timestamp}] CLOSE ({reason}) PnL=${pnl:.2f}")
        
        self.current_trade = None

    def _calculate_pnl(self, trade: Trade) -> float:
        """Calculate trade PnL based on spread trading logic."""
        if trade.side == "LONG_S1_SHORT_S2":
            # LONG S1 SHORT S2 = SELL ALT1, BUY ALT2
            pnl_alt1 = -trade.size_alt1 * (trade.exit_price_alt1 - trade.entry_price_alt1)
            pnl_alt2 = trade.size_alt2 * (trade.exit_price_alt2 - trade.entry_price_alt2)
        else:  # SHORT_S1_LONG_S2
            # SHORT S1 LONG S2 = BUY ALT1, SELL ALT2
            pnl_alt1 = trade.size_alt1 * (trade.exit_price_alt1 - trade.entry_price_alt1)
            pnl_alt2 = -trade.size_alt2 * (trade.exit_price_alt2 - trade.entry_price_alt2)
        
        return pnl_alt1 + pnl_alt2

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
