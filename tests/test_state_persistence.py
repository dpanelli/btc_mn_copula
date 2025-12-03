
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from src.trading import TradingManager
from src.copula_model import SpreadPair

class TestStatePersistence(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_state_manager = MagicMock()
        
        # Mock load_state to return None (simulating missing state file)
        self.mock_state_manager.load_state.return_value = None
        
        self.trading_manager = TradingManager(
            binance_client=self.mock_client,
            capital_per_leg=100.0,
            max_leverage=1,
            state_manager=self.mock_state_manager
        )
        
    def test_save_trade_entry_with_missing_state(self):
        print("\n--- Testing _save_trade_entry with missing state ---")
        
        # Call _save_trade_entry
        self.trading_manager._save_trade_entry(200.0)
        
        # Verify save_state was called with a dict containing trade_entry_time
        self.mock_state_manager.save_state.assert_called_once()
        saved_state = self.mock_state_manager.save_state.call_args[0][0]
        
        self.assertIsInstance(saved_state, dict)
        self.assertIn('trade_entry_time', saved_state)
        self.assertEqual(saved_state['trade_entry_capital'], 200.0)
        print("SUCCESS: _save_trade_entry handled None state correctly!")

    def test_clear_trade_entry_with_missing_state(self):
        print("\n--- Testing _clear_trade_entry with missing state ---")
        
        # Call _clear_trade_entry
        self.trading_manager._clear_trade_entry()
        
        # Verify save_state was NOT called (nothing to clear)
        self.mock_state_manager.save_state.assert_not_called()
        print("SUCCESS: _clear_trade_entry handled None state correctly!")

if __name__ == '__main__':
    unittest.main()
