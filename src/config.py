"""Configuration management for the trading bot."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class BinanceConfig:
    """Binance API configuration."""

    api_key: str
    api_secret: str
    use_testnet: bool


@dataclass
class TradingConfig:
    """Trading strategy configuration."""

    capital_per_leg: float
    max_leverage: int
    entry_threshold: float
    exit_threshold: float
    altcoins: List[str]
    formation_days: int
    trading_days: int
    trading_interval_minutes: int


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    formation_day_of_week: str
    formation_hour: int
    formation_minute: int


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_level: str
    log_file: str
    log_max_bytes: int
    log_backup_count: int


@dataclass
class TelegramConfig:
    """Telegram notification configuration."""

    bot_token: str
    chat_id: str
    enabled: bool


@dataclass
class Config:
    """Main application configuration."""

    binance: BinanceConfig
    trading: TradingConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig
    telegram: TelegramConfig
    state_file: str


def load_config() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config object with all settings

    Raises:
        ValueError: If required environment variables are missing
    """
    # Binance API configuration
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError(
            "BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file"
        )

    binance_config = BinanceConfig(
        api_key=api_key,
        api_secret=api_secret,
        use_testnet=os.getenv("USE_TESTNET", "true").lower() == "true",
    )

    # Trading configuration
    altcoins_str = os.getenv("ALTCOINS", "ETHUSDT,BNBUSDT,ADAUSDT,XRPUSDT,SOLUSDT,AVAXUSDT")
    altcoins = [coin.strip() for coin in altcoins_str.split(",")]

    trading_config = TradingConfig(
        capital_per_leg=float(os.getenv("CAPITAL_PER_LEG", "20000")),
        max_leverage=int(os.getenv("MAX_LEVERAGE", "3")),
        entry_threshold=float(os.getenv("ENTRY_THRESHOLD", "0.10")),
        exit_threshold=float(os.getenv("EXIT_THRESHOLD", "0.10")),
        altcoins=altcoins,
        formation_days=int(os.getenv("FORMATION_DAYS", "21")),
        trading_days=int(os.getenv("TRADING_DAYS", "7")),
        trading_interval_minutes=int(os.getenv("TRADING_INTERVAL_MINUTES", "5")),
    )

    # Scheduler configuration
    scheduler_config = SchedulerConfig(
        formation_day_of_week=os.getenv("FORMATION_DAY_OF_WEEK", "mon"),
        formation_hour=int(os.getenv("FORMATION_HOUR", "0")),
        formation_minute=int(os.getenv("FORMATION_MINUTE", "0")),
    )

    # Logging configuration
    logging_config = LoggingConfig(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/trading.log"),
        log_max_bytes=int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024))),
        log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
    )

    # Telegram configuration (optional)
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_enabled = bool(telegram_bot_token and telegram_chat_id)

    telegram_config = TelegramConfig(
        bot_token=telegram_bot_token,
        chat_id=telegram_chat_id,
        enabled=telegram_enabled,
    )

    # State file
    state_file = os.getenv("STATE_FILE", "state/state.json")

    return Config(
        binance=binance_config,
        trading=trading_config,
        scheduler=scheduler_config,
        logging=logging_config,
        telegram=telegram_config,
        state_file=state_file,
    )


# Global configuration instance
config: Config = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config object

    Raises:
        ValueError: If configuration hasn't been loaded
    """
    global config
    if config is None:
        config = load_config()
    return config
