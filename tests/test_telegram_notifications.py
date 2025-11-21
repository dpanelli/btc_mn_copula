
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies
sys.modules["binance"] = MagicMock()
sys.modules["binance.client"] = MagicMock()
sys.modules["binance.enums"] = MagicMock()
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["statsmodels"] = MagicMock()
sys.modules["statsmodels.tsa.stattools"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["httpx"] = MagicMock()

from src.trading import TradingManager
from src.telegram_notifier import TelegramNotifier

class TestTelegramNotifications(unittest.TestCase):
    def setUp(self):
        self.mock_binance = MagicMock()
        self.mock_telegram = MagicMock()
        
        self.trading_manager = TradingManager(
            binance_client=self.mock_binance,
            capital_per_leg=100,
            max_leverage=1,
            telegram_notifier=self.mock_telegram
        )
        
        # Mock copula model
        self.mock_spread_pair = MagicMock()
        self.mock_spread_pair.alt1 = "XLMUSDT"
        self.mock_spread_pair.alt2 = "ADAUSDT"
        
        self.mock_copula_model = MagicMock()
        self.mock_copula_model.spread_pair = self.mock_spread_pair
        self.mock_copula_model.get_position_quantities.return_value = {
            "XLMUSDT": ("SELL", 100),
            "ADAUSDT": ("BUY", 100)
        }
        
        self.trading_manager.copula_model = self.mock_copula_model

    def test_order_notification_on_entry(self):
        # Mock balance and price
        self.mock_binance.get_account_balance.return_value = 1000.0
        self.mock_binance.get_current_price.return_value = 10.0
        self.mock_binance.calculate_position_size.return_value = 10.0
        
        # Mock order response
        self.mock_binance.place_market_order.return_value = {
            "orderId": "12345",
            "avgPrice": "10.0",
            "symbol": "XLMUSDT"
        }

        # Execute entry
        self.trading_manager._execute_entry_signal("LONG_S1_SHORT_S2")

        # Verify notification called twice (once for each leg)
        self.assertEqual(self.mock_telegram.send_order_notification.call_count, 2)
        
        # Check arguments for first call
        args, kwargs = self.mock_telegram.send_order_notification.call_args_list[0]
        self.assertEqual(kwargs['symbol'], 'XLMUSDT')
        self.assertEqual(kwargs['side'], 'SELL')
        self.assertEqual(kwargs['reduce_only'], False)

    def test_order_notification_on_close(self):
        # Mock close position response
        self.mock_binance.close_position.return_value = {
            "orderId": "67890",
            "avgPrice": "10.0",
            "symbol": "XLMUSDT",
            "side": "BUY",
            "origQty": "10.0"
        }

        # Execute close
        self.trading_manager._close_positions()

        # Verify notification called twice
        self.assertEqual(self.mock_telegram.send_order_notification.call_count, 2)
        
        # Check arguments
        args, kwargs = self.mock_telegram.send_order_notification.call_args_list[0]
        self.assertEqual(kwargs['reduce_only'], True)

if __name__ == '__main__':
    unittest.main()
