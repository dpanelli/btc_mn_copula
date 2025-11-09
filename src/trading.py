"""Trading execution manager for copula-based strategy."""

from datetime import datetime
from typing import Dict, Optional

from .binance_client import BinanceClient
from .copula_model import CopulaModel, SpreadPair
from .logger import get_logger
from .telegram_notifier import TelegramNotifier

logger = get_logger(__name__)


class TradingManager:
    """Manages trading execution based on copula signals."""

    def __init__(
        self,
        binance_client: BinanceClient,
        capital_per_leg: float,
        max_leverage: int,
        entry_threshold: float = 0.10,
        exit_threshold: float = 0.10,
        telegram_notifier: Optional[TelegramNotifier] = None,
    ):
        """
        Initialize trading manager.

        Args:
            binance_client: BinanceClient instance
            capital_per_leg: Capital to allocate per leg in USDT
            max_leverage: Maximum leverage to use
            entry_threshold: Entry threshold α1 (default 0.10)
            exit_threshold: Exit threshold α2 (default 0.10)
            telegram_notifier: Optional TelegramNotifier instance for notifications
        """
        self.binance_client = binance_client
        self.capital_per_leg = capital_per_leg
        self.max_leverage = max_leverage
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.telegram_notifier = telegram_notifier
        self.copula_model: Optional[CopulaModel] = None
        self.current_position: Optional[str] = None  # Track current position state
        self.btc_symbol = "BTCUSDT"

    def set_spread_pair(self, spread_pair: SpreadPair) -> None:
        """
        Set the spread pair to trade and initialize copula model.

        Args:
            spread_pair: SpreadPair object with fitted parameters
        """
        self.copula_model = CopulaModel(
            spread_pair=spread_pair,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
        )

        # Set leverage for both symbols
        try:
            self.binance_client.set_leverage(spread_pair.alt1, self.max_leverage)
            self.binance_client.set_leverage(spread_pair.alt2, self.max_leverage)
            logger.info(
                f"Set leverage to {self.max_leverage}x for {spread_pair.alt1} and {spread_pair.alt2}"
            )
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            raise

    def execute_trading_cycle(self) -> Dict:
        """
        Execute one trading cycle: fetch prices, generate signal, execute trades.

        Returns:
            Dict with execution results
        """
        if self.copula_model is None:
            logger.info("No copula model set - waiting for formation phase to complete")
            return {"status": "waiting", "message": "No copula model - waiting for formation"}

        logger.info("-" * 80)
        logger.info(f"TRADING CYCLE: {datetime.utcnow().isoformat()}")
        logger.info("-" * 80)

        # Log current positions and PnL at start of each cycle
        self._log_positions_and_pnl()
        logger.info("")  # Blank line for readability

        try:
            # Fetch current prices
            btc_price = self.binance_client.get_current_price(self.btc_symbol)
            alt1_price = self.binance_client.get_current_price(
                self.copula_model.spread_pair.alt1
            )
            alt2_price = self.binance_client.get_current_price(
                self.copula_model.spread_pair.alt2
            )

            logger.info(
                f"Current prices: BTC={btc_price:.2f}, "
                f"{self.copula_model.spread_pair.alt1}={alt1_price:.4f}, "
                f"{self.copula_model.spread_pair.alt2}={alt2_price:.4f}"
            )

            # Generate signal
            signal = self.copula_model.generate_signal(btc_price, alt1_price, alt2_price)

            logger.info(f"Signal: {signal}")

            # Send Telegram notification
            if self.telegram_notifier:
                try:
                    positions = self.get_current_positions()

                    # Calculate total PnL
                    total_pnl = sum(
                        pos.get("unrealized_pnl", 0)
                        for pos in positions.values()
                        if "error" not in pos
                    )

                    # Prepare price data
                    price_data = {
                        "btc": btc_price,
                        "alt1": alt1_price,
                        "alt2": alt2_price,
                        "alt1_symbol": self.copula_model.spread_pair.alt1,
                        "alt2_symbol": self.copula_model.spread_pair.alt2,
                    }

                    self.telegram_notifier.send_trading_update(
                        positions=positions,
                        prices=price_data,
                        signal=signal,
                        total_pnl=total_pnl,
                    )
                except Exception as e:
                    logger.error(f"Error sending Telegram notification: {e}")

            # Check if we need to act on the signal
            if signal == "HOLD":
                logger.info("Signal is HOLD - no action taken")
                return {
                    "status": "success",
                    "signal": signal,
                    "action": "none",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Execute trades based on signal
            if signal == "CLOSE":
                result = self._close_positions()
                # Verify positions are closed by querying Binance
                if not self._has_open_positions():
                    self.current_position = None
                else:
                    logger.warning("Positions not fully closed after CLOSE signal")
            elif signal in ["LONG_S1_SHORT_S2", "SHORT_S1_LONG_S2"]:
                # Query Binance for actual positions (avoid using local state)
                has_positions = self._has_open_positions()

                if has_positions:
                    current_position_type = self._get_current_position_type()

                    if current_position_type == signal:
                        # Same position already open - ignore duplicate signal
                        logger.info(
                            f"Position {signal} already open (verified with Binance), "
                            f"ignoring duplicate entry signal"
                        )
                        return {
                            "status": "success",
                            "signal": signal,
                            "action": "none",
                            "message": "Position already open",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    else:
                        # Different position - close old one first
                        logger.info(
                            f"Already in position {current_position_type} (verified with Binance), "
                            f"closing before entering new one"
                        )
                        self._close_positions()
                        # Verify closed
                        if not self._has_open_positions():
                            self.current_position = None
                        else:
                            logger.warning("Failed to close positions before entering new trade")

                # Only execute entry if no position exists
                result = self._execute_entry_signal(signal)
                # Verify entry succeeded by querying Binance
                if result["status"] == "success":
                    if self._has_open_positions():
                        self.current_position = signal
                    else:
                        logger.error("Entry reported success but no positions detected on Binance")
                        self.current_position = None
            else:
                logger.warning(f"Unknown signal: {signal}")
                return {
                    "status": "error",
                    "message": f"Unknown signal: {signal}",
                }

            return result

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _execute_entry_signal(self, signal: str) -> Dict:
        """
        Execute entry signal by opening positions.

        Args:
            signal: Entry signal ('LONG_S1_SHORT_S2' or 'SHORT_S1_LONG_S2')

        Returns:
            Dict with execution results
        """
        positions = self.copula_model.get_position_quantities(
            signal, self.capital_per_leg
        )

        results = {}
        for symbol, (side, capital) in positions.items():
            try:
                # Calculate position size
                position_size = self.binance_client.calculate_position_size(
                    symbol, capital, self.max_leverage
                )

                # Place order
                order = self.binance_client.place_market_order(symbol, side, position_size)

                results[symbol] = {
                    "status": "success",
                    "side": side,
                    "quantity": position_size,
                    "order_id": order.get("orderId"),
                }

                logger.info(
                    f"Opened position: {side} {position_size} {symbol} "
                    f"(order_id={order.get('orderId')})"
                )

            except Exception as e:
                logger.error(f"Error executing {side} order for {symbol}: {e}")
                results[symbol] = {
                    "status": "error",
                    "message": str(e),
                }

        return {
            "status": "success" if all(r.get("status") == "success" for r in results.values()) else "partial",
            "signal": signal,
            "action": "entry",
            "orders": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _close_positions(self) -> Dict:
        """
        Close all open positions.

        Returns:
            Dict with execution results
        """
        if self.copula_model is None:
            return {"status": "error", "message": "No copula model"}

        symbols = [
            self.copula_model.spread_pair.alt1,
            self.copula_model.spread_pair.alt2,
        ]

        results = {}
        for symbol in symbols:
            try:
                order = self.binance_client.close_position(symbol)
                if order:
                    results[symbol] = {
                        "status": "success",
                        "order_id": order.get("orderId"),
                    }
                    logger.info(f"Closed position for {symbol}")
                else:
                    results[symbol] = {
                        "status": "success",
                        "message": "No position to close",
                    }
                    logger.info(f"No position to close for {symbol}")

            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
                results[symbol] = {
                    "status": "error",
                    "message": str(e),
                }

        return {
            "status": "success" if all(r.get("status") == "success" for r in results.values()) else "partial",
            "signal": "CLOSE",
            "action": "close",
            "orders": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_current_positions(self) -> Dict:
        """
        Get current position information for both symbols.

        Returns:
            Dict with position info for each symbol
        """
        if self.copula_model is None:
            return {}

        positions = {}
        for symbol in [
            self.copula_model.spread_pair.alt1,
            self.copula_model.spread_pair.alt2,
        ]:
            try:
                pos = self.binance_client.get_position(symbol)
                positions[symbol] = pos if pos else {"position_amt": 0}
            except Exception as e:
                logger.error(f"Error fetching position for {symbol}: {e}")
                positions[symbol] = {"error": str(e)}

        return positions

    def get_account_info(self) -> Dict:
        """
        Get account balance and position information.

        Returns:
            Dict with account info
        """
        try:
            balance = self.binance_client.get_account_balance()
            positions = self.get_current_positions()

            return {
                "balance_usdt": balance,
                "positions": positions,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {"error": str(e)}

    def _has_open_positions(self) -> bool:
        """
        Check if any positions are open on Binance (queries actual positions).

        Returns:
            True if any position exists, False otherwise
        """
        if self.copula_model is None:
            return False

        positions = self.get_current_positions()
        for symbol, pos_data in positions.items():
            if "error" not in pos_data and pos_data.get("position_amt", 0) != 0:
                return True
        return False

    def _get_current_position_type(self) -> Optional[str]:
        """
        Determine current position type from Binance positions.

        Returns:
            'LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2', or None
        """
        if self.copula_model is None:
            return None

        positions = self.get_current_positions()
        alt1_pos = positions.get(self.copula_model.spread_pair.alt1, {})
        alt2_pos = positions.get(self.copula_model.spread_pair.alt2, {})

        alt1_amt = alt1_pos.get("position_amt", 0)
        alt2_amt = alt2_pos.get("position_amt", 0)

        # Determine position type based on signs
        if alt1_amt > 0 and alt2_amt < 0:
            return "LONG_S1_SHORT_S2"
        elif alt1_amt < 0 and alt2_amt > 0:
            return "SHORT_S1_LONG_S2"
        elif alt1_amt == 0 and alt2_amt == 0:
            return None
        else:
            logger.warning(
                f"Inconsistent positions: {self.copula_model.spread_pair.alt1}={alt1_amt}, "
                f"{self.copula_model.spread_pair.alt2}={alt2_amt}"
            )
            return None

    def _log_positions_and_pnl(self) -> None:
        """
        Log detailed position and PnL information from Binance.
        Logs each position (symbol, size, side, PnL) and total account PnL.
        """
        if self.copula_model is None:
            logger.info("Position Status: No active pair (waiting for formation)")
            return

        try:
            # Get positions from Binance
            positions = self.get_current_positions()

            logger.info("=" * 60)
            logger.info("CURRENT POSITIONS & PnL")
            logger.info("=" * 60)

            total_unrealized_pnl = 0.0
            has_any_position = False

            # Log each leg
            for symbol in [self.copula_model.spread_pair.alt1, self.copula_model.spread_pair.alt2]:
                pos_data = positions.get(symbol, {})

                if "error" in pos_data:
                    logger.error(f"{symbol}: Error fetching position - {pos_data['error']}")
                    continue

                position_amt = pos_data.get("position_amt", 0)

                if position_amt == 0:
                    logger.info(f"{symbol}: FLAT (no position)")
                else:
                    has_any_position = True
                    entry_price = pos_data.get("entry_price", 0)
                    unrealized_pnl = pos_data.get("unrealized_pnl", 0)
                    side = "LONG" if position_amt > 0 else "SHORT"

                    # Calculate notional position size
                    notional = abs(position_amt) * entry_price

                    logger.info(
                        f"{symbol}: {side} | Size: {abs(position_amt):.4f} | "
                        f"Entry: ${entry_price:.4f} | Notional: ${notional:.2f} | "
                        f"Unrealized PnL: ${unrealized_pnl:+.2f}"
                    )

                    total_unrealized_pnl += unrealized_pnl

            # Log total PnL
            logger.info("-" * 60)
            if has_any_position:
                logger.info(f"TOTAL UNREALIZED PnL: ${total_unrealized_pnl:+.2f}")

                # Also log local state vs reality
                local_state = self.current_position
                binance_state = self._get_current_position_type()
                if local_state != binance_state:
                    logger.warning(
                        f"STATE MISMATCH: Local={local_state}, Binance={binance_state}"
                    )
            else:
                logger.info("TOTAL UNREALIZED PnL: $0.00 (no open positions)")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error logging positions and PnL: {e}", exc_info=True)
