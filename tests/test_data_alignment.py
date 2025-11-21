
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Mock dependencies BEFORE importing src
sys.modules["binance"] = MagicMock()
sys.modules["binance.client"] = MagicMock()
sys.modules["binance.enums"] = MagicMock()  # This was missing/incorrectly handled
sys.modules["scipy"] = MagicMock()
sys.modules["scipy.stats"] = MagicMock()
sys.modules["statsmodels"] = MagicMock()
sys.modules["statsmodels.tsa.stattools"] = MagicMock()

from src.formation import FormationManager

class TestDataAlignment(unittest.TestCase):
    def setUp(self):
        self.mock_binance = MagicMock()
        self.formation_manager = FormationManager(
            binance_client=self.mock_binance,
            altcoins=["ALT1", "ALT2"],
            formation_days=1
        )

    @patch("src.formation.calculate_spread")
    @patch("src.formation.check_cointegration")
    @patch("src.formation.kendalltau")
    @patch("src.copula_model.kendalltau")  # Mock it here too!
    def test_alignment_with_missing_data(self, mock_tau_copula, mock_tau_formation, mock_coint, mock_spread):
        # Setup mocks
        mock_spread.return_value = (np.array([1, 2, 3]), 1.0)
        mock_coint.return_value = True
        mock_tau_formation.return_value = (0.5, 0.01)
        mock_tau_copula.return_value = (0.5, 0.01)  # Return tuple for copula model

        # Generate 200 timestamps
        timestamps = [pd.Timestamp("2023-01-01 10:00") + pd.Timedelta(minutes=5*i) for i in range(200)]
        
        # BTC has all 200 candles
        btc_df = pd.DataFrame({
            "timestamp": timestamps,
            "close": [100.0 + i for i in range(200)]
        })

        # ALT1 is missing index 10 (timestamp[10])
        alt1_timestamps = timestamps.copy()
        alt1_timestamps.pop(10)
        alt1_df = pd.DataFrame({
            "timestamp": alt1_timestamps,
            "close": [10.0 + i for i in range(199)]
        })

        # ALT2 is missing index 20 (timestamp[20])
        alt2_timestamps = timestamps.copy()
        alt2_timestamps.pop(20)
        alt2_df = pd.DataFrame({
            "timestamp": alt2_timestamps,
            "close": [20.0 + i for i in range(199)]
        })

        # Mock Binance responses
        def get_klines(symbol, *args):
            if symbol == "BTCUSDT": return btc_df
            if symbol == "ALT1": return alt1_df
            if symbol == "ALT2": return alt2_df
            return pd.DataFrame()

        self.mock_binance.get_historical_klines.side_effect = get_klines

        # Run formation
        self.formation_manager.run_formation()

        # Verify calculate_spread was called with ALIGNED data
        # We expect 198 elements (200 total - 1 missing from ALT1 - 1 missing from ALT2)
        
        # Check calls
        self.assertTrue(mock_spread.called)
        args, _ = mock_spread.call_args_list[0]
        btc_prices = args[0]
        alt_prices = args[1]

        print(f"Aligned BTC length: {len(btc_prices)}")
        print(f"Aligned ALT length: {len(alt_prices)}")

        # Should have 198 elements
        self.assertEqual(len(btc_prices), 198)
        self.assertEqual(len(alt_prices), 198)

if __name__ == '__main__':
    unittest.main()
