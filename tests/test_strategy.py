import pytest
from src.strategy import PairsTradingStrategy
from src.copula_model import SpreadPair
import numpy as np

@pytest.fixture
def strategy():
    pair = SpreadPair(alt1="ETHUSDT", alt2="BNBUSDT")
    pair.beta1 = 0.5
    pair.beta2 = 0.5
    # Add dummy data for CopulaModel initialization
    pair.spread1_data = np.random.normal(0, 1, 100)
    pair.spread2_data = np.random.normal(0, 1, 100)
    pair.rho = 0.8
    return PairsTradingStrategy(spread_pair=pair, capital_per_leg=1000.0)

def test_get_target_positions_long_s1(strategy):
    """Verify LONG_S1_SHORT_S2 results in Buy Alt1 / Sell Alt2"""
    prices = {"ETHUSDT": 2000.0, "BNBUSDT": 300.0}
    positions = strategy.get_target_positions("LONG_S1_SHORT_S2", prices)
    
    assert positions["ETHUSDT"][0] == "BUY"
    assert positions["BNBUSDT"][0] == "SELL"
    assert positions["ETHUSDT"][1] == 1000.0 / 2000.0  # 0.5
    assert positions["BNBUSDT"][1] == 1000.0 / 300.0   # 3.33...

def test_get_target_positions_short_s1(strategy):
    """Verify SHORT_S1_LONG_S2 results in Sell Alt1 / Buy Alt2"""
    prices = {"ETHUSDT": 2000.0, "BNBUSDT": 300.0}
    positions = strategy.get_target_positions("SHORT_S1_LONG_S2", prices)
    
    assert positions["ETHUSDT"][0] == "SELL"
    assert positions["BNBUSDT"][0] == "BUY"

def test_get_position_state(strategy):
    """Verify state detection from positions"""
    # Long Alt1, Short Alt2 -> LONG_S1_SHORT_S2
    assert strategy.get_position_state({"ETHUSDT": 0.5, "BNBUSDT": -3.3}) == "LONG_S1_SHORT_S2"
    
    # Short Alt1, Long Alt2 -> SHORT_S1_LONG_S2
    assert strategy.get_position_state({"ETHUSDT": -0.5, "BNBUSDT": 3.3}) == "SHORT_S1_LONG_S2"
    
    # Flat
    assert strategy.get_position_state({"ETHUSDT": 0, "BNBUSDT": 0}) is None
    
    # Inconsistent (Long/Long)
    assert strategy.get_position_state({"ETHUSDT": 0.5, "BNBUSDT": 3.3}) == "INCONSISTENT"
    
    # Inconsistent (One leg only)
    assert strategy.get_position_state({"ETHUSDT": 0.5, "BNBUSDT": 0}) == "INCONSISTENT"

def test_calculate_pnl(strategy):
    """Verify PnL calculation logic"""
    entry_prices = {"ETHUSDT": 2000.0, "BNBUSDT": 300.0}
    quantities = {"ETHUSDT": 0.5, "BNBUSDT": 3.3333}
    
    # Scenario 1: LONG S1 (Long ETH, Short BNB)
    # ETH goes up 10%, BNB goes down 10% -> PROFIT on both
    exit_prices = {"ETHUSDT": 2200.0, "BNBUSDT": 270.0}
    
    pnl = strategy.calculate_pnl(entry_prices, exit_prices, quantities, "LONG_S1_SHORT_S2")
    
    expected_eth_pnl = 0.5 * (2200 - 2000) # 100
    expected_bnb_pnl = -3.3333 * (270 - 300) # -3.3333 * -30 = 100
    
    assert abs(pnl - (expected_eth_pnl + expected_bnb_pnl)) < 0.01
    assert pnl > 0
    
    # Scenario 2: SHORT S1 (Short ETH, Long BNB)
    # ETH goes up, BNB goes down -> LOSS on both
    pnl_loss = strategy.calculate_pnl(entry_prices, exit_prices, quantities, "SHORT_S1_LONG_S2")
    assert pnl_loss < 0
