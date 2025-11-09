"""Unit tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from src.config import BinanceConfig, Config, TradingConfig, load_config


class TestConfiguration:
    """Test configuration loading."""

    def test_load_config_with_env_vars(self):
        """Test loading configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "test_key",
                "BINANCE_API_SECRET": "test_secret",
                "USE_TESTNET": "true",
                "CAPITAL_PER_LEG": "15000",
                "MAX_LEVERAGE": "2",
                "ENTRY_THRESHOLD": "0.15",
                "EXIT_THRESHOLD": "0.12",
            },
        ):
            config = load_config()

            assert isinstance(config, Config)
            assert config.binance.api_key == "test_key"
            assert config.binance.api_secret == "test_secret"
            assert config.binance.use_testnet is True
            assert config.trading.capital_per_leg == 15000
            assert config.trading.max_leverage == 2
            assert config.trading.entry_threshold == 0.15
            assert config.trading.exit_threshold == 0.12

    def test_load_config_defaults(self):
        """Test loading configuration with default values."""
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "test_key",
                "BINANCE_API_SECRET": "test_secret",
            },
            clear=True,
        ):
            config = load_config()

            # Check defaults
            assert config.trading.capital_per_leg == 20000.0
            assert config.trading.max_leverage == 3
            assert config.trading.entry_threshold == 0.10
            assert config.trading.exit_threshold == 0.10

    def test_load_config_missing_api_keys(self):
        """Test that missing API keys raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="BINANCE_API_KEY"):
                load_config()

    def test_altcoins_parsing(self):
        """Test parsing altcoins from comma-separated string."""
        with patch.dict(
            os.environ,
            {
                "BINANCE_API_KEY": "test_key",
                "BINANCE_API_SECRET": "test_secret",
                "ALTCOINS": "ETHUSDT,BNBUSDT,ADAUSDT",
            },
        ):
            config = load_config()

            assert config.trading.altcoins == ["ETHUSDT", "BNBUSDT", "ADAUSDT"]
