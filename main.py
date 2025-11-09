"""Main orchestrator for BTC Market-Neutral Copula Trading Bot."""

import signal
import sys
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.binance_client import BinanceClient
from src.config import get_config
from src.formation import FormationManager
from src.logger import get_logger, setup_logger
from src.state_manager import StateManager
from src.telegram_notifier import TelegramNotifier
from src.trading import TradingManager

# Global instances
logger = None
binance_client = None
formation_manager = None
trading_manager = None
state_manager = None
telegram_notifier = None
scheduler = None
is_formation_running = False  # Lock to prevent trading during formation


def initialize_components():
    """Initialize all components with configuration."""
    global logger, binance_client, formation_manager, trading_manager, state_manager, telegram_notifier

    # Load configuration
    config = get_config()

    # Set up logger
    logger = setup_logger(
        name="trading_bot",
        log_file=config.logging.log_file,
        log_level=config.logging.log_level,
        max_bytes=config.logging.log_max_bytes,
        backup_count=config.logging.log_backup_count,
    )

    logger.info("=" * 100)
    logger.info("BTC MARKET-NEUTRAL COPULA TRADING BOT")
    logger.info("=" * 100)
    logger.info(f"Started at: {datetime.utcnow().isoformat()} UTC")
    logger.info(f"Testnet mode: {config.binance.use_testnet}")
    logger.info(f"Capital per leg: ${config.trading.capital_per_leg:,.2f}")
    logger.info(f"Max leverage: {config.trading.max_leverage}x")
    logger.info(f"Entry threshold (α1): {config.trading.entry_threshold}")
    logger.info(f"Exit threshold (α2): {config.trading.exit_threshold}")
    logger.info(f"Altcoins: {', '.join(config.trading.altcoins)}")
    logger.info(f"Formation: {config.trading.formation_days} days")
    logger.info(f"Trading interval: {config.trading.trading_interval_minutes} minutes")
    logger.info("=" * 100)

    # Initialize Binance client
    binance_client = BinanceClient(
        api_key=config.binance.api_key,
        api_secret=config.binance.api_secret,
        testnet=config.binance.use_testnet,
    )

    # Initialize formation manager
    formation_manager = FormationManager(
        binance_client=binance_client,
        altcoins=config.trading.altcoins,
        formation_days=config.trading.formation_days,
        interval="5m",  # Use 5-minute candles as per instructions
    )

    # Initialize Telegram notifier (optional)
    telegram_notifier = None
    if config.telegram.enabled:
        telegram_notifier = TelegramNotifier(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id,
            enabled=True,
        )
        logger.info("Telegram notifications enabled")
    else:
        logger.info("Telegram notifications disabled")

    # Initialize trading manager
    trading_manager = TradingManager(
        binance_client=binance_client,
        capital_per_leg=config.trading.capital_per_leg,
        max_leverage=config.trading.max_leverage,
        entry_threshold=config.trading.entry_threshold,
        exit_threshold=config.trading.exit_threshold,
        telegram_notifier=telegram_notifier,
    )

    # Initialize state manager
    state_manager = StateManager(state_file=config.state_file)

    logger.info("All components initialized successfully")

    # Try to load existing formation state
    spread_pair = state_manager.load_formation_state()
    if spread_pair:
        logger.info("Loaded existing formation state, setting up trading manager")
        trading_manager.set_spread_pair(spread_pair)
        # Restore position state
        current_position = state_manager.get_current_position()
        trading_manager.current_position = current_position
        logger.info(f"Current position: {current_position}")
    else:
        logger.warning(
            "No existing formation state found. "
            "Formation phase will run on next scheduled time or manually."
        )


def run_formation_phase():
    """Execute formation phase: select pairs and fit copula."""
    global formation_manager, trading_manager, state_manager, logger, is_formation_running

    try:
        # Set formation lock to prevent trading during formation
        is_formation_running = True

        logger.info("\n" + "=" * 100)
        logger.info("STARTING WEEKLY FORMATION PHASE")
        logger.info("=" * 100)

        # Close any existing positions before formation
        if trading_manager.copula_model is not None:
            logger.info("Closing existing positions before formation...")
            trading_manager._close_positions()
            trading_manager.current_position = None

        # Run formation
        spread_pair = formation_manager.run_formation()

        if spread_pair is None:
            logger.error("Formation phase failed - no suitable pairs found!")
            return

        # Save formation state
        state_manager.save_formation_state(spread_pair)

        # Update trading manager with new pair
        trading_manager.set_spread_pair(spread_pair)
        trading_manager.current_position = None

        logger.info("=" * 100)
        logger.info("FORMATION PHASE COMPLETED SUCCESSFULLY")
        logger.info(f"Selected pair: {spread_pair.alt1} - {spread_pair.alt2}")
        logger.info(f"Parameters: β1={spread_pair.beta1:.6f}, β2={spread_pair.beta2:.6f}, ρ={spread_pair.rho:.4f}")
        logger.info("=" * 100 + "\n")

    except Exception as e:
        logger.error(f"Error in formation phase: {e}", exc_info=True)
    finally:
        # Always release formation lock
        is_formation_running = False


def run_trading_cycle():
    """Execute trading cycle: generate signals and execute trades."""
    global trading_manager, state_manager, logger, is_formation_running

    # Skip trading if formation is running
    if is_formation_running:
        logger.debug("Skipping trading cycle - formation phase in progress")
        return

    try:
        # Execute trading cycle
        result = trading_manager.execute_trading_cycle()

        # Log trade result
        if result.get("status") != "error":
            state_manager.save_trade_log(result)

        # Update position state if changed
        if result.get("action") in ["entry", "close"]:
            state_manager.update_position_state(trading_manager.current_position)

    except Exception as e:
        logger.error(f"Error in trading cycle: {e}", exc_info=True)


def shutdown_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global scheduler, logger

    logger.info("\n" + "=" * 100)
    logger.info("SHUTDOWN SIGNAL RECEIVED")
    logger.info("=" * 100)

    if scheduler:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown(wait=False)

    logger.info("Shutdown complete")
    sys.exit(0)


def main():
    """Main entry point for the trading bot."""
    global scheduler

    # Initialize components
    initialize_components()

    # Auto-formation: Run formation immediately if no state exists
    # This ensures the bot can start trading right away
    if state_manager.load_formation_state() is None:
        logger.info("\n" + "=" * 100)
        logger.info("NO FORMATION STATE FOUND - RUNNING INITIAL FORMATION")
        logger.info("=" * 100)
        run_formation_phase()

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Get configuration
    config = get_config()

    # Create scheduler
    scheduler = BlockingScheduler()

    # Schedule formation phase (weekly on configured day and time)
    formation_trigger = CronTrigger(
        day_of_week=config.scheduler.formation_day_of_week,
        hour=config.scheduler.formation_hour,
        minute=config.scheduler.formation_minute,
        timezone="UTC",
    )
    scheduler.add_job(
        run_formation_phase,
        trigger=formation_trigger,
        id="formation_phase",
        name="Weekly Formation Phase",
        replace_existing=True,
    )
    logger.info(
        f"Scheduled formation phase: Every {config.scheduler.formation_day_of_week} "
        f"at {config.scheduler.formation_hour:02d}:{config.scheduler.formation_minute:02d} UTC"
    )

    # Schedule trading cycles (synchronized to clock: 00:02, 05:02, 10:02, etc.)
    # Run 2 seconds after each 5-minute mark to ensure candles are closed
    trading_trigger = CronTrigger(
        minute=f"*/{config.trading.trading_interval_minutes}",
        second="2",
        timezone="UTC",
    )
    scheduler.add_job(
        run_trading_cycle,
        trigger=trading_trigger,
        id="trading_cycle",
        name=f"{config.trading.trading_interval_minutes}-minute Trading Cycle (synchronized)",
        replace_existing=True,
    )
    logger.info(
        f"Scheduled trading cycles: Every {config.trading.trading_interval_minutes} minutes "
        f"(synchronized to clock: XX:00:02, XX:05:02, XX:10:02, etc.)"
    )

    # Print next scheduled runs
    logger.info("\nNext scheduled jobs:")
    for job in scheduler.get_jobs():
        # Use trigger.get_next_fire_time() since scheduler hasn't started yet
        next_run = job.trigger.get_next_fire_time(None, datetime.now(job.trigger.timezone))
        logger.info(f"  - {job.name}: {next_run}")

    logger.info("\n" + "=" * 100)
    logger.info("BOT IS NOW RUNNING")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 100 + "\n")

    # Start scheduler (blocking)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
