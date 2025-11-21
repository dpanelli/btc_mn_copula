"""Test atomic order placement with rollback on partial fills."""

import pytest
from unittest.mock import Mock, patch
from src.trading import TradingManager
from src.copula_model import SpreadPair
import numpy as np


def test_partial_fill_triggers_rollback():
    """Test that partial fills trigger automatic rollback of successful orders."""
    
    # Setup
    mock_client = Mock()
    mock_client.get_account_balance.return_value = 50000.0
    
    # Simulate partial fill scenario:
    # LONG_S1_SHORT_S2 means: SELL S1 (ETHUSDT), BUY S2 (BNBUSDT)
    # - First order (ETHUSDT SELL) succeeds
    # - Second order (BNBUSDT BUY) fails
    order_counter = [0]
    
    def place_market_order_side_effect(symbol, side, quantity):
        order_counter[0] += 1
        
        if symbol == "ETHUSDT" and order_counter[0] == 1:
            # First order succeeds (SELL)
            assert side == "SELL", f"First order should be SELL, got {side}"
            return {
                "orderId": 12345,
                "symbol": symbol,
                "status": "FILLED",
                "executedQty": quantity,
            }
        elif symbol == "BNBUSDT" and order_counter[0] == 2:
            # Second order fails (e.g., rate limit) - should be BUY
            assert side == "BUY", f"Second order should be BUY, got {side}"
            raise Exception("APIError(code=-1008): Request throttled")
        elif order_counter[0] == 3:
            # Rollback order for ETHUSDT (should be BUY to close SELL)
            assert side == "BUY", f"Rollback should be BUY (opposite of SELL), got {side}"
            assert symbol == "ETHUSDT", "Should rollback ETHUSDT"
            return {
                "orderId": 12346,
                "symbol": symbol,
                "status": "FILLED",
                "executedQty": quantity,
            }
        
        raise Exception("Unexpected order")
    
    mock_client.place_market_order.side_effect = place_market_order_side_effect
    mock_client.calculate_position_size.return_value = 100.0
    mock_client.get_current_price.return_value = 3000.0
    
    # Create trading manager
    trading_manager = TradingManager(
        binance_client=mock_client,
        capital_per_leg=20000,
        max_leverage=3,
    )
    
    # Set spread pair
    spread_pair = SpreadPair("ETHUSDT", "BNBUSDT")
    spread_pair.beta1 = 0.5
    spread_pair.beta2 = 200.0
    spread_pair.rho = 0.6
    spread_pair.tau = 0.4
    spread_pair.spread1_data = np.random.normal(0, 100, 1000)
    spread_pair.spread2_data = np.random.normal(0, 50, 1000)
    trading_manager.set_spread_pair(spread_pair)
    
    # Execute entry signal
    result = trading_manager._execute_entry_signal("LONG_S1_SHORT_S2")
    
    # Verify rollback was triggered
    assert result["status"] == "error", "Partial fill should return error status"
    assert result["action"] == "entry_failed_with_rollback", "Should indicate rollback occurred"
    assert "rollback" in result, "Should contain rollback results"
    assert len(result["rollback"]) > 0, "Should have rolled back at least one order"
    assert result["rollback"]["ETHUSDT"]["status"] == "success", "Rollback should succeed"
    
    # Verify 3 orders were placed: 2 entry attempts + 1 rollback
    assert order_counter[0] == 3, "Should have placed 3 orders total (2 entry + 1 rollback)"
    
    print("✓ Partial fill rollback test PASSED")
    print(f"  - First order (ETHUSDT SELL) succeeded")
    print(f"  - Second order (BNBUSDT BUY) failed")
    print(f"  - Automatic rollback (ETHUSDT BUY) triggered")
    print(f"  - Result: No inconsistent position ✓")


def test_all_orders_succeed_no_rollback():
    """Test that successful orders don't trigger rollback."""
    
    # Setup
    mock_client = Mock()
    mock_client.get_account_balance.return_value = 50000.0
    
    # Both orders succeed
    def place_market_order_success(symbol, side, quantity):
        return {
            "orderId": 123,
            "symbol": symbol,
            "status": "FILLED",
            "executedQty": quantity,
        }
    
    mock_client.place_market_order.side_effect = place_market_order_success
    mock_client.calculate_position_size.return_value = 100.0
    mock_client.get_current_price.return_value = 3000.0
    
    # Create trading manager
    trading_manager = TradingManager(
        binance_client=mock_client,
        capital_per_leg=20000,
        max_leverage=3,
    )
    
    # Set spread pair
    spread_pair = SpreadPair("ETHUSDT", "BNBUSDT")
    spread_pair.beta1 = 0.5
    spread_pair.beta2 = 200.0
    spread_pair.rho = 0.6
    spread_pair.tau = 0.4
    spread_pair.spread1_data = np.random.normal(0, 100, 1000)
    spread_pair.spread2_data = np.random.normal(0, 50, 1000)
    trading_manager.set_spread_pair(spread_pair)
    
    # Execute entry signal
    result = trading_manager._execute_entry_signal("LONG_S1_SHORT_S2")
    
    # Verify success without rollback
    assert result["status"] == "success", "All successful orders should return success"
    assert "rollback" not in result, "No rollback should occur on success"
    assert len(result["orders"]) == 2, "Should have 2 successful orders"
    
    # Only 2 orders should be placed (no rollback)
    assert mock_client.place_market_order.call_count == 2, "Should only place 2 orders"
    
    print("✓ Successful orders test PASSED")
    print(f"  - Both orders succeeded")
    print(f"  - No rollback triggered")
    print(f"  - Result: Success ✓")


if __name__ == "__main__":
    test_partial_fill_triggers_rollback()
    test_all_orders_succeed_no_rollback()
    print("\n✅ ALL ROLLBACK TESTS PASSED!")
