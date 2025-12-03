
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from src.trading import TradingManager
from src.copula_model import SpreadPair

class TestReentryBug(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_state_manager = MagicMock()
        
        # Mock load_state to return empty dict initially
        self.mock_state_manager.load_state.return_value = {}
        
        self.trading_manager = TradingManager(
            binance_client=self.mock_client,
            capital_per_leg=100.0,
            max_leverage=1,
            entry_threshold=0.1,
            exit_threshold=0.1,
            stop_loss_pct=0.04,
            state_manager=self.mock_state_manager,
            cooldown_minutes=60
        )
        
        # Setup a dummy strategy/pair
        pair = SpreadPair("ETHUSDT", "BNBUSDT")
        pair.beta1 = 1.0
        pair.beta2 = 1.0
        pair.rho = 0.9
        pair.spread1_data = [0.0] * 100 # Dummy history
        pair.spread2_data = [0.0] * 100
        
        # Mock the strategy to force signals
        self.trading_manager.set_spread_pair(pair)
        self.trading_manager.strategy = MagicMock()
        self.trading_manager.strategy.spread_pair = pair
        
    def test_reentry_after_stop_loss(self):
        # 1. Setup: Position is OPEN
        self.trading_manager.current_position = "LONG_S1_SHORT_S2"
        
        # Mock has_open_positions to return True initially
        self.trading_manager._has_open_positions = MagicMock(return_value=True)
        self.trading_manager._get_current_position_type = MagicMock(return_value="LONG_S1_SHORT_S2")
        
        # Mock PnL to trigger Stop Loss (-50%)
        self.trading_manager._get_position_pnl = MagicMock(return_value=-50.0) 
        
        # Mock Close Positions
        self.trading_manager._close_positions = MagicMock(return_value={"status": "success"})
        self.trading_manager._clear_trade_entry = MagicMock()
        
        # Mock Prices
        self.mock_client.get_current_price.return_value = 100.0
        
        # EXECUTE CYCLE 1: Should trigger Stop Loss AND Activate Cooldown
        print("\n--- Cycle 1: Check Stop Loss ---")
        result1 = self.trading_manager.execute_trading_cycle()
        print(f"Result 1: {result1}")
        
        self.assertEqual(result1['status'], 'stop_loss')
        self.trading_manager._close_positions.assert_called()
        self.mock_state_manager.update_cooldown.assert_called() # Verify cooldown update called
        
        # 2. Setup: Position is CLOSED (after stop loss)
        self.trading_manager._has_open_positions = MagicMock(return_value=False)
        self.trading_manager.current_position = None
        
        # Mock Strategy to return ENTRY signal again
        self.trading_manager.strategy.generate_signal.return_value = MagicMock(
            signal="LONG_S1_SHORT_S2", 
            metadata={}
        )
        
        # Mock Entry execution
        self.trading_manager._execute_entry_signal = MagicMock(return_value={"status": "success"})
        
        # SIMULATE COOLDOWN STATE
        # The state manager should now return a cooldown timestamp in the future
        future_time = (datetime.now(timezone.utc) + timedelta(minutes=60)).isoformat()
        self.mock_state_manager.load_state.return_value = {'cooldown_until': future_time}
        
        # EXECUTE CYCLE 2: Should return WAITING due to cooldown
        print("\n--- Cycle 2: Next 5 mins (Cooldown Active) ---")
        result2 = self.trading_manager.execute_trading_cycle()
        print(f"Result 2: {result2}")
        
        # Assertion: Should be waiting, NOT entry
        self.assertEqual(result2.get('status'), 'waiting')
        self.assertIn("Cooldown active", result2.get('message', ''))
        self.trading_manager._execute_entry_signal.assert_not_called()
        print("SUCCESS: Bot correctly waited due to cooldown!")

if __name__ == '__main__':
    unittest.main()
