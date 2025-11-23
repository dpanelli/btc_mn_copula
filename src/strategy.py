"""
Centralized strategy logic for Pairs Trading.
This module encapsulates all trading rules, signal generation, and state detection logic.
It is designed to be STATELESS for live trading, meaning it takes current state as input
and returns decisions/signals as output.
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np

from .copula_model import CopulaModel, SpreadPair
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class TradeSignal:
    signal: str  # 'LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2', 'CLOSE', 'HOLD'
    reason: str
    metadata: Dict

class PairsTradingStrategy:
    """
    Centralized strategy logic.
    
    Responsibilities:
    1. Signal Generation (wrapping CopulaModel)
    2. Position Sizing (calculating exact quantities)
    3. State Detection (mapping Binance positions to Strategy state)
    4. PnL Calculation (standardized math for backtesting/simulation)
    """
    
    def __init__(
        self,
        spread_pair: SpreadPair,
        entry_threshold: float = 0.10,
        exit_threshold: float = 0.10,
        capital_per_leg: float = 1000.0,
        max_leverage: int = 1
    ):
        self.spread_pair = spread_pair
        self.capital_per_leg = capital_per_leg
        self.max_leverage = max_leverage
        
        # Initialize underlying model
        self.model = CopulaModel(
            spread_pair=spread_pair,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold
        )
        
    def generate_signal(self, btc_price: float, alt1_price: float, alt2_price: float) -> TradeSignal:
        """
        Generate trading signal based on current prices.
        """
        result = self.model.generate_signal(btc_price, alt1_price, alt2_price)
        return TradeSignal(
            signal=result['signal'],
            reason=f"h_1|2={result.get('h1_2', 0):.4f}, h_2|1={result.get('h2_1', 0):.4f}",
            metadata=result
        )
        
    def get_target_positions(self, signal: str, prices: Dict[str, float]) -> Dict[str, Tuple[str, float]]:
        """
        Calculate target position sizes for a given signal.
        
        Args:
            signal: 'LONG_S1_SHORT_S2' or 'SHORT_S1_LONG_S2'
            prices: Dict of {symbol: price}
            
        Returns:
            Dict of {symbol: (side, quantity)}
            e.g. {'ETHUSDT': ('BUY', 1.5), 'BTCUSDT': ('SELL', 0.1)}
        """
        if signal == "LONG_S1_SHORT_S2":
            # LONG S1 = LONG ALT1 / SHORT ALT2
            # S1 = ALT1 - beta * BTC. Long S1 means betting S1 goes UP.
            # So we BUY ALT1.
            # S2 = ALT2 - beta * BTC. Short S2 means betting S2 goes DOWN.
            # So we SELL ALT2.
            return {
                self.spread_pair.alt1: ("BUY", self.capital_per_leg / prices[self.spread_pair.alt1]),
                self.spread_pair.alt2: ("SELL", self.capital_per_leg / prices[self.spread_pair.alt2])
            }
        elif signal == "SHORT_S1_LONG_S2":
            # SHORT S1 = SHORT ALT1 / LONG ALT2
            return {
                self.spread_pair.alt1: ("SELL", self.capital_per_leg / prices[self.spread_pair.alt1]),
                self.spread_pair.alt2: ("BUY", self.capital_per_leg / prices[self.spread_pair.alt2])
            }
        else:
            return {}

    def get_position_state(self, positions: Dict[str, float]) -> Optional[str]:
        """
        Determine current strategy state from open positions.
        
        Args:
            positions: Dict of {symbol: signed_quantity}
                       Positive = Long, Negative = Short
                       
        Returns:
            'LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2', 'INCONSISTENT', or None (Flat)
        """
        alt1_qty = positions.get(self.spread_pair.alt1, 0)
        alt2_qty = positions.get(self.spread_pair.alt2, 0)
        
        # Tolerance for floating point dust
        if abs(alt1_qty) < 1e-6 and abs(alt2_qty) < 1e-6:
            return None
            
        # LONG S1 = Long Alt1 / Short Alt2
        if alt1_qty > 0 and alt2_qty < 0:
            return "LONG_S1_SHORT_S2"
            
        # SHORT S1 = Short Alt1 / Long Alt2
        if alt1_qty < 0 and alt2_qty > 0:
            return "SHORT_S1_LONG_S2"
            
        # Anything else is inconsistent (e.g. Long/Long, Short/Short, or one leg only)
        return "INCONSISTENT"

    def calculate_pnl(
        self, 
        entry_prices: Dict[str, float], 
        exit_prices: Dict[str, float], 
        quantities: Dict[str, float],
        state: str
    ) -> float:
        """
        Calculate PnL for a trade.
        PURELY FOR BACKTESTING/SIMULATION. Live PnL should come from exchange.
        
        Args:
            entry_prices: {symbol: price}
            exit_prices: {symbol: price}
            quantities: {symbol: abs_quantity}
            state: 'LONG_S1_SHORT_S2' or 'SHORT_S1_LONG_S2'
        """
        pnl = 0.0
        
        alt1 = self.spread_pair.alt1
        alt2 = self.spread_pair.alt2
        
        qty1 = quantities.get(alt1, 0)
        qty2 = quantities.get(alt2, 0)
        
        if state == "LONG_S1_SHORT_S2":
            # Long Alt1, Short Alt2
            pnl += qty1 * (exit_prices[alt1] - entry_prices[alt1])
            pnl += -qty2 * (exit_prices[alt2] - entry_prices[alt2])
        elif state == "SHORT_S1_LONG_S2":
            # Short Alt1, Long Alt2
            pnl += -qty1 * (exit_prices[alt1] - entry_prices[alt1])
            pnl += qty2 * (exit_prices[alt2] - entry_prices[alt2])
            
        return pnl
