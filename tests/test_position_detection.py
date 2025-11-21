"""Test that position type detection correctly matches signal-to-order mapping."""

import pytest
from unittest.mock import Mock
from src.trading import TradingManager
from src.copula_model import SpreadPair
import numpy as np


def test_short_s1_long_s2_position_detection():
    """Test SHORT_S1_LONG_S2 signal creates correct position type."""
    
    # Setup
    mock_client = Mock()
    trading_manager = TradingManager(
        binance_client=mock_client,
        capital_per_leg=20000,
        max_leverage=3,
    )
    
    # Set spread pair
    spread_pair = SpreadPair("ADAUSDT", "AVAXUSDT")
    spread_pair.beta1 = 0.5
    spread_pair.beta2 = 200.0
    spread_pair.rho = 0.6
    spread_pair.tau = 0.4
    spread_pair.spread1_data = np.random.normal(0, 100, 1000)
    spread_pair.spread2_data = np.random.normal(0, 50, 1000)
    trading_manager.set_spread_pair(spread_pair)
    
    # Simulate positions after executing SHORT_S1_LONG_S2:
    # - Signal places: BUY ADAUSDT, SELL AVAXUSDT
    # - Results in: LONG ADAUSDT (+), SHORT AVAXUSDT (-)
    mock_client.get_position.side_effect = [
        {"position_amt": 250.0},   # ADAUSDT: LONG (positive)
        {"position_amt": -8.0},    # AVAXUSDT: SHORT (negative)
    ]
    
    # Get detected position type
    detected_type = trading_manager._get_current_position_type()
    
    # Should correctly identify as SHORT_S1_LONG_S2
    assert detected_type == "SHORT_S1_LONG_S2", (
        f"Expected 'SHORT_S1_LONG_S2' but got '{detected_type}'. "
        f"Signal SHORT_S1_LONG_S2 places BUY ALT1 SELL ALT2, "
        f"creating LONG ALT1 (+250) SHORT ALT2 (-8) positions."
    )
    
    print("✓ SHORT_S1_LONG_S2 detection CORRECT")
    print(f"  - ADAUSDT: +250.0 (LONG)")
    print(f"  - AVAXUSDT: -8.0 (SHORT)")
    print(f"  - Detected as: {detected_type} ✓")


def test_long_s1_short_s2_position_detection():
    """Test LONG_S1_SHORT_S2 signal creates correct position type."""
    
    # Setup
    mock_client = Mock()
    trading_manager = TradingManager(
        binance_client=mock_client,
        capital_per_leg=20000,
        max_leverage=3,
    )
    
    # Set spread pair
    spread_pair = SpreadPair("ADAUSDT", "AVAXUSDT")
    spread_pair.beta1 = 0.5
    spread_pair.beta2 = 200.0
    spread_pair.rho = 0.6
    spread_pair.tau = 0.4
    spread_pair.spread1_data = np.random.normal(0, 100, 1000)
    spread_pair.spread2_data = np.random.normal(0, 50, 1000)
    trading_manager.set_spread_pair(spread_pair)
    
    # Simulate positions after executing LONG_S1_SHORT_S2:
    # - Signal places: SELL ADAUSDT, BUY AVAXUSDT
    # - Results in: SHORT ADAUSDT (-), LONG AVAXUSDT (+)
    mock_client.get_position.side_effect = [
        {"position_amt": -250.0},  # ADAUSDT: SHORT (negative)
        {"position_amt": 8.0},     # AVAXUSDT: LONG (positive)
    ]
    
    # Get detected position type
    detected_type = trading_manager._get_current_position_type()
    
    # Should correctly identify as LONG_S1_SHORT_S2
    assert detected_type == "LONG_S1_SHORT_S2", (
        f"Expected 'LONG_S1_SHORT_S2' but got '{detected_type}'. "
        f"Signal LONG_S1_SHORT_S2 places SELL ALT1 BUY ALT2, "
        f"creating SHORT ALT1 (-250) LONG ALT2 (+8) positions."
    )
    
    print("✓ LONG_S1_SHORT_S2 detection CORRECT")
    print(f"  - ADAUSDT: -250.0 (SHORT)")
    print(f"  - AVAXUSDT: +8.0 (LONG)")
    print(f"  - Detected as: {detected_type} ✓")


def test_no_position_flipping():
    """Test that signal and detected position match (no flipping)."""
    
    # Setup
    mock_client = Mock()
    trading_manager = TradingManager(
        binance_client=mock_client,
        capital_per_leg=20000,
        max_leverage=3,
    )
    
    # Set spread pair
    spread_pair = SpreadPair("ADAUSDT", "AVAXUSDT")
    spread_pair.beta1 = 0.5
    spread_pair.beta2 = 200.0
    spread_pair.rho = 0.6
    spread_pair.tau = 0.4
    spread_pair.spread1_data = np.random.normal(0, 100, 1000)
    spread_pair.spread2_data = np.random.normal(0, 50, 1000)
    trading_manager.set_spread_pair(spread_pair)
    
    # Test both signals
    test_cases = [
        {
            "signal": "SHORT_S1_LONG_S2",
            "orders": [("BUY", "ADAUSDT"), ("SELL", "AVAXUSDT")],
            "positions": [250.0, -8.0],  # LONG ADA, SHORT AVAX
            "expected": "SHORT_S1_LONG_S2",
        },
        {
            "signal": "LONG_S1_SHORT_S2",
            "orders": [("SELL", "ADAUSDT"), ("BUY", "AVAXUSDT")],
            "positions": [-250.0, 8.0],  # SHORT ADA, LONG AVAX
            "expected": "LONG_S1_SHORT_S2",
        },
    ]
    
    for case in test_cases:
        # Get orders that would be placed for this signal
        orders = trading_manager.copula_model.get_position_quantities(
            case["signal"], 20000
        )
        
        # Verify orders match expected
        alt1_order = orders["ADAUSDT"][0]
        alt2_order = orders["AVAXUSDT"][0]
        assert alt1_order == case["orders"][0][0], f"ALT1 order mismatch for {case['signal']}"
        assert alt2_order == case["orders"][1][0], f"ALT2 order mismatch for {case['signal']}"
        
        # Simulate those positions
        mock_client.get_position.side_effect = [
            {"position_amt": case["positions"][0]},
            {"position_amt": case["positions"][1]},
        ]
        
        # Verify detection matches original signal
        detected = trading_manager._get_current_position_type()
        assert detected == case["expected"], (
            f"Signal {case['signal']} should be detected as {case['expected']}, "
            f"but got {detected}. This would cause position flipping!"
        )
        
        print(f"✓ {case['signal']} → {case['orders']} → {detected} ✓")
    
    print("\n✅ NO POSITION FLIPPING - Signal/Position mapping is correct!")


if __name__ == "__main__":
    test_short_s1_long_s2_position_detection()
    test_long_s1_short_s2_position_detection()
    test_no_position_flipping()
    print("\n✅ ALL POSITION DETECTION TESTS PASSED!")
