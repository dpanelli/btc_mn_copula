
import unittest
from unittest.mock import MagicMock
from src.trading import TradingManager
from src.copula_model import SpreadPair

class TestDoubleEntry(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.trading_manager = TradingManager(
            binance_client=self.mock_client,
            capital_per_leg=100.0,
            max_leverage=1
        )
        
        # Setup dummy strategy
        pair = SpreadPair("ETHUSDT", "BNBUSDT")
        pair.spread1_data = [0.0] * 100
        pair.spread2_data = [0.0] * 100
        pair.beta1 = 1.0
        pair.beta2 = 1.0
        pair.rho = 0.5
        self.trading_manager.set_spread_pair(pair)
        
    def test_has_open_positions_on_error(self):
        print("\n--- Testing _has_open_positions with API Error ---")
        
        # Mock get_current_positions to return an error for one symbol
        # This simulates the behavior I saw in the code:
        # positions[symbol] = {"error": str(e)}
        self.trading_manager.get_current_positions = MagicMock(return_value={
            "ETHUSDT": {"error": "API Timeout"},
            "BNBUSDT": {"position_amt": 0}
        })
        
        # Call _has_open_positions
        # It SHOULD return True (or raise error) to be safe.
        # But I suspect it returns False because it ignores keys with "error".
        has_positions = self.trading_manager._has_open_positions()
        
        print(f"Result: {has_positions}")
        
        # If it returns False, it means the bot would proceed to open a new position!
        if not has_positions:
            print("BUG REPRODUCED: Bot thinks it has NO positions despite API error!")
        else:
            print("Behavior is safe.")
            
        # Assert that it now returns True (FAIL SAFE)
        self.assertTrue(has_positions)
        print("SUCCESS: _has_open_positions returned True on error (Safe Behavior)!")

if __name__ == '__main__':
    unittest.main()
