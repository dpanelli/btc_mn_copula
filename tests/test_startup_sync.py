"""Test that startup correctly syncs with Binance instead of using stale state file."""

from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_startup_uses_binance_position_not_stale_file():
    """Test that startup queries Binance for positions, not state file."""
    
    with patch('main.BinanceClient') as MockBinanceClient, \
         patch('main.FormationManager') as MockFormationManager, \
         patch('main.TradingManager') as MockTradingManager, \
         patch('main.StateManager') as MockStateManager, \
         patch('main.TelegramNotifier') as MockTelegramNotifier:
        
        # Mock state manager
        mock_state_mgr = MagicMock()
        MockStateManager.return_value = mock_state_mgr
        
        # Mock spread pair loaded from file
        mock_spread_pair = MagicMock()
        mock_spread_pair.alt1 = "ADAUSDT"
        mock_spread_pair.alt2 = "AVAXUSDT"
        mock_state_mgr.load_formation_state.return_value = mock_spread_pair
        
        # STALE FILE DATA says position is SHORT_S1_LONG_S2
        mock_state_mgr.get_current_position.return_value = "SHORT_S1_LONG_S2"
        
        # Mock trading manager
        mock_trading_mgr = MagicMock()
        MockTradingManager.return_value = mock_trading_mgr
        
        # BINANCE REALITY says no positions (flat)
        mock_trading_mgr._get_current_position_type.return_value = None
        
        # Import and run initialization
        import main
        main.initialize()
        
        # Verify it called Binance query, not used file data
        mock_trading_mgr._get_current_position_type.assert_called_once()
        
        # Verify it set position to Binance reality (None), not file data
        assert mock_trading_mgr.current_position is None, \
            "Should use Binance position (None), not stale file (SHORT_S1_LONG_S2)"
        
        print("✓ Startup correctly queries Binance")
        print(f"  - State file said: SHORT_S1_LONG_S2 (STALE)")
        print(f"  - Binance said: None (flat)")
        print(f"  - Bot set position to: None ✅")


if __name__ == "__main__":
    test_startup_uses_binance_position_not_stale_file()
    print("\n✅ STARTUP SYNC TEST PASSED!")
