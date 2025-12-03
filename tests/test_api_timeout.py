
import unittest
from unittest.mock import MagicMock, patch
from src.trading import TradingManager
from src.copula_model import SpreadPair

class TestAPITimeout(unittest.TestCase):
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
        
    def test_get_current_position_type_with_api_error(self):
        print("\n--- Testing _get_current_position_type with API Error ---")
        
        # Mock get_position to fail for one symbol
        def side_effect(symbol):
            if symbol == "ETHUSDT":
                return {"position_amt": "1.0"}
            else:
                raise Exception("API Timeout")
                
        self.mock_client.get_position.side_effect = side_effect
        
        # Call _get_current_position_type
        # It should catch the exception, log error, and return None (NOT "INCONSISTENT")
        position_type = self.trading_manager._get_current_position_type()
        
        print(f"Result: {position_type}")
        
        # Verify it returns None (safe fallback) instead of INCONSISTENT
        self.assertIsNone(position_type)
        print("SUCCESS: Handled API error gracefully (returned None)!")

    @patch('time.sleep', return_value=None) # Mock sleep to speed up test
    def test_retry_logic(self, mock_sleep):
        print("\n--- Testing Retry Logic ---")
        
        # Mock to fail twice then succeed
        self.mock_client.get_position.side_effect = [
            Exception("Fail 1"),
            Exception("Fail 2"),
            {"position_amt": "1.0"}
        ]
        
        # Call get_current_positions for ONE symbol (to simplify test)
        # We need to hack the loop in get_current_positions or just test the method
        # Let's test get_current_positions directly
        
        # We need to reset side_effect for the loop
        # The loop calls get_position for alt1 then alt2
        # Let's make alt1 fail twice then succeed, and alt2 succeed immediately
        self.mock_client.get_position.side_effect = [
            Exception("Fail 1"),
            Exception("Fail 2"),
            {"position_amt": "1.0"}, # Alt1 success (3rd try)
            {"position_amt": "0.5"}  # Alt2 success (1st try)
        ]
        
        positions = self.trading_manager.get_current_positions(max_retries=3)
        
        self.assertIn("ETHUSDT", positions)
        self.assertNotIn("error", positions["ETHUSDT"])
        self.assertEqual(positions["ETHUSDT"]["position_amt"], "1.0")
        
        print("SUCCESS: Retry logic worked!")

if __name__ == '__main__':
    unittest.main()
