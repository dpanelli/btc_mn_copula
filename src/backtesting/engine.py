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
        
        # Mock FormationManager to use our DataManager
        self.formation_manager = FormationManager(
            binance_client=self.dm.client, # We'll patch this or use DM directly
            altcoins=config.trading.altcoins,
            formation_days=config.trading.formation_days
        )
        # Monkey patch formation manager to use DM instead of direct client calls if needed
        # For now, we assume DM populates the cache, but FormationManager uses client.
        # Ideally, FormationManager should accept a data_provider interface.
        # To keep it simple, we will let FormationManager use the client (which might hit API or cache if we wrap it).
        # BUT, for backtesting speed, we really want it to use local data.
        # We will override the get_historical_klines method of the client passed to FormationManager.
        
        self.formation_manager.binance_client.get_historical_klines = self._mock_get_klines

    def _mock_get_klines(self, symbol, interval, start, end):
        """Redirect FormationManager calls to DataManager."""
        return self.dm.get_data(symbol, start, end, interval)

    def run(self):
        """Run the backtest simulation."""
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
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
            
        # Temporarily override current time for the formation manager logic?
        # FormationManager calculates start_time based on datetime.now(datetime.UTC).
        # We need to subclass or patch it to respect 'current_time'.
        # Actually, FormationManager takes start/end in run_formation? No, it calculates internally.
        # We need to patch datetime.utcnow or modify FormationManager.
        # Let's modify FormationManager instance to accept an 'end_time' argument in run_formation
        # OR, since we can't easily change the class without editing the file, we'll just 
        # manually call the logic here using our data.
        
        # RE-IMPLEMENTING FORMATION LOGIC SIMPLIFIED FOR BACKTEST
        # (Or better: Refactor FormationManager to accept end_time. 
        #  I will assume I can patch it or I'll just copy the logic for now to be safe).
        
        # Let's try to use the existing manager but patch the time.
        # This is hacky but effective for backtesting without refactoring everything.
        with np.testing.suppress_warnings() as sup:
            # We need to trick FormationManager into thinking 'now' is 'current_time'
            # Since we can't easily mock datetime.utcnow inside the module from here without complex patching,
            # I will add a helper method to FormationManager in a separate step if needed.
            # For now, let's assume I can pass end_time to a modified run_formation.
            pass

        # ACTUALLY: I will modify FormationManager in the next step to accept 'current_time'.
        # It's a small change that makes it testable.
        
        # For now, let's assume it works and returns a pair.
        # I will implement the 'run_formation_at' method in FormationManager in the next step.
        
        # Placeholder:
        # pair = self.formation_manager.run_formation(end_time=current_time)
        pass 

    def _process_cycle(self, current_time: datetime):
        """Process a single trading cycle."""
        # Get current prices
        prices = self._get_current_prices(current_time)
        if not prices:
            return

        # Update model prices
        # CopulaModel needs latest prices to calculate h-values
        # We need to feed it the latest data.
        # The CopulaModel.generate_signal uses self.binance_client.get_current_price
        # We need to patch that too!
        
        # ... Implementation details ...
        pass

    def _update_equity(self, current_time: datetime):
        """Track equity curve."""
        unrealized_pnl = 0
        if self.current_trade:
            # Calculate mark-to-market PnL
            pass
            
        self.equity_curve.append({
            "timestamp": current_time,
            "equity": self.equity + unrealized_pnl,
            "cash": self.equity
        })

    def _get_current_prices(self, timestamp: datetime) -> Dict[str, float]:
        """Get prices for all active assets at timestamp."""
        return {}
