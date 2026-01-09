
import unittest
from unittest.mock import MagicMock
from src.trading import TradingManager
from src.copula_model import SpreadPair

class TestLeftovers(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_state_manager = MagicMock()
        self.mock_state_manager.load_state.return_value = {} # Return dict, not Mock
        self.trading_manager = TradingManager(
            binance_client=self.mock_client,
            capital_per_leg=100.0,
            max_leverage=1,
            state_manager=self.mock_state_manager
        )
        
        # Setup dummy strategy
        pair = SpreadPair("ETHUSDT", "BNBUSDT")
        pair.spread1_data = [0.0] * 100
        pair.spread2_data = [0.0] * 100
        pair.beta1 = 1.0
        pair.beta2 = 1.0
        pair.rho = 0.5
        self.trading_manager.set_spread_pair(pair)
        
    def test_close_positions_failure(self):
        print("\n--- Testing _close_positions failure ---")
        
        # Mock close_position to fail for one symbol
        def side_effect(symbol):
            if symbol == "ETHUSDT":
                return {"orderId": "123", "status": "FILLED"}
            else:
                raise Exception("API Error")
                
        self.mock_client.close_position.side_effect = side_effect
        
        # Call _close_positions
        result = self.trading_manager._close_positions()
        
        # Verify result is partial
        self.assertEqual(result["status"], "partial")
        
        # Verify trade entry was NOT cleared (save_state not called with removal)
        # Actually _clear_trade_entry calls load_state then save_state.
        # Since we mocked state_manager, we can check if save_state was called.
        # But wait, _clear_trade_entry logic:
        # state = load_state()
        # state.pop(...)
        # save_state(state)
        
        # If we didn't call _clear_trade_entry, save_state shouldn't be called (assuming no other saves)
        self.mock_state_manager.save_state.assert_not_called()
        print("SUCCESS: Trade entry was NOT cleared on partial close!")

    def test_abort_entry_on_close_failure(self):
        print("\n--- Testing abort entry on close failure ---")
        
        # Mock has_open_positions to return True (simulating stuck position)
        self.trading_manager._has_open_positions = MagicMock(return_value=True)
        self.trading_manager._get_current_position_type = MagicMock(return_value="LONG_S1_SHORT_S2")
        self.trading_manager._get_position_pnl = MagicMock(return_value=0.0) # No loss
        
        # Mock _close_positions to return partial success (failed)
        self.trading_manager._close_positions = MagicMock(return_value={"status": "partial"})
        
        # Mock strategy to generate a DIFFERENT signal (triggering close-then-open)
        mock_signal = MagicMock()
        mock_signal.signal = "SHORT_S1_LONG_S2"
        mock_signal.metadata = {}
        self.trading_manager.strategy.generate_signal = MagicMock(return_value=mock_signal)
        
        # Mock prices
        self.mock_client.get_current_price.return_value = 100.0
        
        # Execute cycle
        result = self.trading_manager.execute_trading_cycle()
        
        # Verify status is error
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Failed to close existing positions")
        
        # Verify _execute_entry_signal was NOT called
        # We can check this by spying on it, but since we didn't mock it, 
        # we can check if binance_client.place_market_order was called for ENTRY.
        # But _close_positions was mocked, so no close orders.
        # If entry happened, place_market_order would be called.
        self.mock_client.place_market_order.assert_not_called()
        
        print("SUCCESS: Entry was aborted when close failed!")

if __name__ == '__main__':
    unittest.main()
