
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock binance module before importing src.trading
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

class TestBalanceCheck(unittest.TestCase):
    def setUp(self):
        self.mock_binance = MagicMock()
        self.trading_manager = TradingManager(
            binance_client=self.mock_binance,
            capital_per_leg=100,  # Need 200 total
            max_leverage=1
        )
        
        # Mock copula model
        self.mock_spread_pair = MagicMock()
        self.mock_spread_pair.alt1 = "XLMUSDT"
        self.mock_spread_pair.alt2 = "ADAUSDT"
        
        self.mock_copula_model = MagicMock()
        self.mock_copula_model.spread_pair = self.mock_spread_pair
        # Return positions requiring capital
        self.mock_copula_model.get_position_quantities.return_value = {
            "XLMUSDT": ("SELL", 100),
            "ADAUSDT": ("BUY", 100)
        }
        
        self.trading_manager.copula_model = self.mock_copula_model

    def test_insufficient_balance(self):
        # Mock balance to be less than required (150 < 200)
        self.mock_binance.get_account_balance.return_value = 150.0

        # Execute entry signal
        result = self.trading_manager._execute_entry_signal("LONG_S1_SHORT_S2")

        # Verify results
        print(f"Result: {result}")
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Insufficient balance")
        
        # Should NOT place orders
        self.mock_binance.place_market_order.assert_not_called()

    def test_sufficient_balance(self):
        # Mock balance to be enough (250 > 200)
        self.mock_binance.get_account_balance.return_value = 250.0
        
        # Mock position size calc
        self.mock_binance.calculate_position_size.return_value = 10.0

        # Execute entry signal
        result = self.trading_manager._execute_entry_signal("LONG_S1_SHORT_S2")

        # Verify results
        self.assertEqual(result["status"], "success")
        
        # Should place orders
        self.assertEqual(self.mock_binance.place_market_order.call_count, 2)

if __name__ == '__main__':
    unittest.main()
