"""Test cooldown mechanism to prevent position churn."""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.trading import TradingManager
from src.copula_model import SpreadPair


def test_cooldown_mechanism():
    """Test that cooldown prevents rapid position flips."""
    
    # Create mock binance client
    mock_client = Mock()
    # Use return_value instead of side_effect to allow unlimited calls
    mock_client.get_current_price = Mock(return_value=100000)
    mock_client.get_position = Mock(return_value=None)
    mock_client.get_account_balance = Mock(return_value=10000.0)  # Sufficient balance
    mock_client.close_position = Mock(return_value={
        "orderId": "12345",
        "side": "SELL",
        "origQty": "1.0",
    })
    mock_client.place_market_order = Mock(return_value={
        "orderId": "12345",
        "avgPrice": "3000",
    })
    mock_client.calculate_position_size = Mock(return_value=1.0)
    
    # Create trading manager with 1 minute cooldown for testing
    trading_manager = TradingManager(
        binance_client=mock_client,
        capital_per_leg=100,
        max_leverage=1,
        cooldown_minutes=1,  # 1 minute cooldown
    )
    
    # Create test spread pair
    spread_pair = SpreadPair("ETHUSDT", "BNBUSDT")
    spread_pair.beta1 = 30.0
    spread_pair.beta2 = 200.0
    spread_pair.rho = 0.5
    spread_pair.tau = 0.3
    spread_pair.spread1_data = [100, 200, 300]
    spread_pair.spread2_data = [50, 100, 150]
    
    trading_manager.set_spread_pair(spread_pair)
    
    # Mock copula model to generate LONG_S1_SHORT_S2 signal
    with patch.object(trading_manager.copula_model, 'generate_signal') as mock_signal:
        mock_signal.return_value = {
            "signal": "LONG_S1_SHORT_S2",
            "h_1_given_2": 0.05,
            "h_2_given_1": 0.95,
            "entry_threshold": 0.10,
            "exit_threshold": 0.10,
            "distance_1": 0.45,
            "distance_2": 0.45,
        }
        
        print("\\n" + "="*80)
        print("TEST: Cooldown Mechanism")
        print("="*80)
        
        # Execute trading cycle - should trigger CLOSE
        print("\\nStep 1: Mock positions exist, signal to close them")
        mock_client.get_position = Mock(return_value={
            "position_amt": 1.0,
            "entry_price": 3000,
            "unrealized_pnl": 10,
        })
        
        # Change signal to CLOSE
        mock_signal.return_value["signal"] = "CLOSE"
        
        result = trading_manager.execute_trading_cycle()
        print(f"  Result: {result.get('action')} - {result.get('message', result.get('status'))}")
        
        # Verify close time was set
        assert trading_manager.last_close_time is not None, "last_close_time should be set after close"
        print(f"  ✓ Close time tracked: {trading_manager.last_close_time}")
        
        # Step 2: Immediately try to enter opposite position (should be blocked by cooldown)
        print("\\nStep 2: Immediately try to enter LONG_S1_SHORT_S2 (should be blocked)")
        mock_client.get_position = Mock(return_value=None)  # No positions
        mock_signal.return_value["signal"] = "LONG_S1_SHORT_S2"
        
        result = trading_manager.execute_trading_cycle()
        print(f"  Result: {result.get('action')} - {result.get('message')}")
        
        # Verify entry was blocked
        assert result["action"] == "cooldown", "Entry should be blocked by cooldown"
        print(f"  ✓ Entry blocked by cooldown")
        
        # Step 3: Simulate time passage (mock datetime.utcnow)
        print("\\nStep 3: Fast-forward 2 minutes (beyond cooldown)")
        future_time = datetime.utcnow() + timedelta(minutes=2)
        
        with patch('src.trading.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = future_time
            
            result = trading_manager.execute_trading_cycle()
            print(f"  Result: {result.get('action')} - {result.get('message', result.get('status'))}")
            
            # Verify entry is now allowed
            assert result["action"] in ["entry", "partial"], "Entry should be allowed after cooldown"
            print(f"  ✓ Entry allowed after cooldown expired")
        
        print("\\n" + "="*80)
        print("COOLDOWN TEST PASSED ✓")
        print("="*80)


if __name__ == "__main__":
    test_cooldown_mechanism()
    print("\\nAll cooldown tests passed!")
