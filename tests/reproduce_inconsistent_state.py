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
from src.copula_model import SpreadPair

class TestInconsistentState(unittest.TestCase):
    def setUp(self):
        self.mock_binance = MagicMock()
        self.trading_manager = TradingManager(
            binance_client=self.mock_binance,
            capital_per_leg=100,
            max_leverage=1
        )
        
        # Mock copula model and spread pair
        self.mock_spread_pair = MagicMock()
        self.mock_spread_pair.alt1 = "XLMUSDT"
        self.mock_spread_pair.alt2 = "ADAUSDT"
        
        self.mock_copula_model = MagicMock()
        self.mock_copula_model.spread_pair = self.mock_spread_pair
        # Mock signal generation to return a trade signal
        self.mock_copula_model.generate_signal.return_value = {"signal": "LONG_S1_SHORT_S2"}
        
        self.trading_manager.copula_model = self.mock_copula_model
        self.trading_manager.btc_symbol = "BTCUSDT"

    def test_inconsistent_state_handling(self):
        # Setup inconsistent positions: XLM short, ADA flat
        # This matches the user's log: XLMUSDT=-872.0, ADAUSDT=0
        self.mock_binance.get_position.side_effect = lambda symbol: {
            "XLMUSDT": {"symbol": "XLMUSDT", "position_amt": -872.0, "entry_price": 0.1, "unrealized_pnl": 0},
            "ADAUSDT": {"symbol": "ADAUSDT", "position_amt": 0.0, "entry_price": 0.5, "unrealized_pnl": 0}
        }.get(symbol)

        # Mock price fetching
        self.mock_binance.get_current_price.return_value = 100.0

        # Execute trading cycle
        result = self.trading_manager.execute_trading_cycle()

        # Verify results
        print(f"Result: {result}")
        
        # 1. Should detect inconsistent state
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["action"], "close_inconsistent")
        
        # 2. Should call close_position for both assets
        self.mock_binance.close_position.assert_any_call("XLMUSDT")
        self.mock_binance.close_position.assert_any_call("ADAUSDT")
        
        # 3. Should NOT call place_market_order (to open new positions)
        # close_position calls place_market_order with reduce_only=True, so we check that NO reduce_only=False calls were made
        for call in self.mock_binance.place_market_order.call_args_list:
            args, kwargs = call
            # kwargs might be in args if positional, but signature is (symbol, side, quantity, reduce_only=False)
            # reduce_only is 4th arg or kwarg
            reduce_only = kwargs.get('reduce_only', False)
            if len(args) > 3:
                reduce_only = args[3]
            
            self.assertTrue(reduce_only, f"Found a market order that was not reduce_only: {call}")

if __name__ == '__main__':
    unittest.main()
