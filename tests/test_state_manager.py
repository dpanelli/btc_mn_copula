"""Unit tests for state manager."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.copula_model import SpreadPair
from src.state_manager import StateManager


class TestStateManager:
    """Test StateManager functionality."""

    @pytest.fixture
    def temp_state_file(self):
        """Create a temporary state file path (file not created yet)."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name
        # Delete the file that was just created
        Path(temp_path).unlink(missing_ok=True)
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def mock_spread_pair(self):
        """Create a mock SpreadPair for testing."""
        pair = SpreadPair("ETHUSDT", "BNBUSDT")
        pair.beta1 = 2.5
        pair.beta2 = 1.8
        pair.rho = 0.6
        pair.tau = 0.45

        # Create mock historical spread data
        np.random.seed(42)
        pair.spread1_data = np.random.randn(100) * 10 + 50
        pair.spread2_data = np.random.randn(100) * 8 + 40

        return pair

    def test_save_formation_state(self, temp_state_file, mock_spread_pair):
        """Test saving formation state."""
        manager = StateManager(state_file=temp_state_file)
        manager.save_formation_state(mock_spread_pair)

        # Check file exists
        assert Path(temp_state_file).exists()

        # Check file contents
        with open(temp_state_file, "r") as f:
            state = json.load(f)

        assert "formation_timestamp" in state
        assert state["pair"]["alt1"] == "ETHUSDT"
        assert state["pair"]["alt2"] == "BNBUSDT"
        assert state["parameters"]["beta1"] == 2.5
        assert state["parameters"]["beta2"] == 1.8
        assert state["parameters"]["rho"] == 0.6
        assert state["parameters"]["tau"] == 0.45
        assert len(state["spread_data"]["spread1"]) == 100
        assert len(state["spread_data"]["spread2"]) == 100

    def test_load_formation_state(self, temp_state_file, mock_spread_pair):
        """Test loading formation state."""
        manager = StateManager(state_file=temp_state_file)

        # Save first
        manager.save_formation_state(mock_spread_pair)

        # Load
        loaded_pair = manager.load_formation_state()

        assert loaded_pair is not None
        assert loaded_pair.alt1 == "ETHUSDT"
        assert loaded_pair.alt2 == "BNBUSDT"
        assert loaded_pair.beta1 == 2.5
        assert loaded_pair.beta2 == 1.8
        assert loaded_pair.rho == 0.6
        assert loaded_pair.tau == 0.45
        assert len(loaded_pair.spread1_data) == 100
        assert len(loaded_pair.spread2_data) == 100

    def test_load_nonexistent_state(self, temp_state_file):
        """Test loading state when file doesn't exist."""
        # Use a non-existent file
        manager = StateManager(state_file=temp_state_file + "_nonexistent")

        loaded_pair = manager.load_formation_state()

        assert loaded_pair is None

    def test_update_position_state(self, temp_state_file, mock_spread_pair):
        """Test updating position state."""
        manager = StateManager(state_file=temp_state_file)

        # Save formation state first
        manager.save_formation_state(mock_spread_pair)

        # Update position
        manager.update_position_state("LONG_S1_SHORT_S2")

        # Load and check
        with open(temp_state_file, "r") as f:
            state = json.load(f)

        assert state["current_position"] == "LONG_S1_SHORT_S2"
        assert "position_updated_at" in state

    def test_get_current_position(self, temp_state_file, mock_spread_pair):
        """Test getting current position."""
        manager = StateManager(state_file=temp_state_file)

        # Save formation state
        manager.save_formation_state(mock_spread_pair)

        # Update position
        manager.update_position_state("SHORT_S1_LONG_S2")

        # Get position
        position = manager.get_current_position()

        assert position == "SHORT_S1_LONG_S2"

    def test_save_trade_log(self, temp_state_file):
        """Test saving trade log."""
        manager = StateManager(state_file=temp_state_file)

        trade_data = {
            "timestamp": "2024-01-01T00:00:00",
            "signal": "LONG_S1_SHORT_S2",
            "status": "success",
        }

        manager.save_trade_log(trade_data)

        # Check trade log file exists
        log_file = Path(temp_state_file).parent / "trade_log.jsonl"
        assert log_file.exists()

        # Check contents
        with open(log_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1
        logged_trade = json.loads(lines[0])
        assert logged_trade["signal"] == "LONG_S1_SHORT_S2"

        # Cleanup
        log_file.unlink()

    def test_get_state_summary(self, temp_state_file, mock_spread_pair):
        """Test getting state summary."""
        manager = StateManager(state_file=temp_state_file)

        # No state yet
        summary = manager.get_state_summary()
        assert summary["status"] == "no_state"

        # Save state
        manager.save_formation_state(mock_spread_pair)

        # Get summary
        summary = manager.get_state_summary()

        assert summary["status"] == "active"
        assert "formation_timestamp" in summary
        assert summary["pair"]["alt1"] == "ETHUSDT"
        assert summary["pair"]["alt2"] == "BNBUSDT"

    def test_clear_state(self, temp_state_file, mock_spread_pair):
        """Test clearing state."""
        manager = StateManager(state_file=temp_state_file)

        # Save state
        manager.save_formation_state(mock_spread_pair)
        assert Path(temp_state_file).exists()

        # Clear state
        manager.clear_state()
        assert not Path(temp_state_file).exists()
