"""Trading execution manager for copula-based strategy."""

from datetime import datetime, timezone
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
        state_manager: Optional[object] = None,
        stop_loss_pct: float = 0.04,
        max_trade_duration_hours: int = 48,
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
            state_manager: Optional StateManager for crash-resistant state storage
            stop_loss_pct: Stop-loss as % of position value (default 0.04 = 4%)
            max_trade_duration_hours: Maximum trade duration in hours (default 48)
        """
        self.binance_client = binance_client
        self.capital_per_leg = capital_per_leg
        self.max_leverage = max_leverage
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.telegram_notifier = telegram_notifier
        self.state_manager = state_manager
        self.stop_loss_pct = stop_loss_pct
        self.max_trade_duration_hours = max_trade_duration_hours
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
        logger.info(f"TRADING CYCLE: {datetime.now(timezone.utc).isoformat()}")
        logger.info("-" * 80)

        # Log current positions and PnL at start of each cycle
        self._log_positions_and_pnl()
        logger.info("")  # Blank line for readability

        # CRITICAL: Check for inconsistent state immediately
        # This ensures we clean up bad states even if the signal is HOLD
        if self._has_open_positions():
            current_position_type = self._get_current_position_type()
            if current_position_type == "INCONSISTENT":
                logger.warning("Detected inconsistent positions. Closing all and waiting for next cycle.")
                self._close_positions()
                return {
                    "status": "warning",
                    "signal": "INCONSISTENT_STATE",
                    "action": "close_inconsistent",
                    "message": "Closed inconsistent positions, waiting for next cycle",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        # Sync local state with Binance reality
        # This prevents "STATE MISMATCH" warnings if the bot was restarted
        real_position = self._get_current_position_type()
        if self.current_position != real_position:
            logger.info(
                f"Syncing local state with Binance: {self.current_position} -> {real_position}"
            )
            self.current_position = real_position

        # RISK MANAGEMENT: Check stop-loss and time-based exit for open positions
        if self._has_open_positions():
            # 1. Stop-loss check (percentage-based)
            unrealized_pnl = self._get_position_pnl()  # From Binance API
            
            # Get entry capital from state, or use default
            entry_capital = self.capital_per_leg * 2  # Default
            if self.state_manager:
                try:
                    state = self.state_manager.load_state()
                    entry_capital = state.get('trade_entry_capital', entry_capital)
                except Exception as e:
                    logger.warning(f"Could not load entry capital from state: {e}")
            
            stop_loss_threshold = -entry_capital * self.stop_loss_pct
            
            if unrealized_pnl < stop_loss_threshold:
                logger.warning(
                    f"🛑 STOP-LOSS triggered: PnL=${unrealized_pnl:.2f} < "
                    f"${stop_loss_threshold:.2f} ({self.stop_loss_pct:.1%} of ${entry_capital:.2f})"
                )
                self._close_positions()
                self._clear_trade_entry()
                
                if self.telegram_notifier:
                    self.telegram_notifier.send_message(
                        f"🛑 STOP-LOSS: Closed positions at ${unrealized_pnl:.2f}"
                    )
                
                return {
                    "status": "stop_loss",
                    "pnl": unrealized_pnl,
                    "threshold": stop_loss_threshold,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            
            # 2. Time-based exit check
            entry_time = self._get_trade_entry_time()
            if entry_time:
                duration_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
                
                if duration_hours > self.max_trade_duration_hours:
                    logger.warning(
                        f"⏰ TIME-BASED EXIT: Trade duration {duration_hours:.1f}h > "
                        f"{self.max_trade_duration_hours}h"
                    )
                    self._close_positions()
                    self._clear_trade_entry()
                    
                    if self.telegram_notifier:
                        self.telegram_notifier.send_message(
                            f"⏰ TIME EXIT: Closed positions after {duration_hours:.1f}h, PnL=${unrealized_pnl:.2f}"
                        )
                    
                    return {
                        "status": "time_exit",
                        "duration_hours": duration_hours,
                        "pnl": unrealized_pnl,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
            else:
                # No entry time found - log warning but continue
                if self._has_open_positions():
                    logger.warning("No entry time found in state, skipping time-based exit check")

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
            signal_data = self.copula_model.generate_signal(btc_price, alt1_price, alt2_price)
            signal = signal_data["signal"]  # Extract signal string for backward compatibility

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
                        signal_data=signal_data,
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                            "timestamp": datetime.now(timezone.utc).isoformat(),
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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

        # Check if we have enough balance for all legs
        required_capital = sum(cap for _, cap in positions.values())
        try:
            available_balance = self.binance_client.get_account_balance()
            if available_balance < required_capital:
                logger.error(
                    f"Insufficient balance for entry: Available={available_balance:.2f}, "
                    f"Required={required_capital:.2f}"
                )
                return {
                    "status": "error",
                    "message": "Insufficient balance",
                    "available": available_balance,
                    "required": required_capital,
                }
        except Exception as e:
            logger.error(f"Error checking balance: {e}")
            # Proceed with caution or return error? 
            # Safer to return error if we can't verify balance
            return {
                "status": "error",
                "message": f"Could not verify balance: {e}",
            }

        results = {}
        successful_orders = []  # Track successful orders for potential rollback

        # Phase 1: Execute all orders
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
                
                # Track for potential rollback
                successful_orders.append({
                    "symbol": symbol,
                    "side": side,
                    "quantity": position_size,
                    "order_id": order.get("orderId"),
                })

                logger.info(
                    f"Opened position: {side} {position_size} {symbol} "
                    f"(order_id={order.get('orderId')})"
                )

                # Send immediate notification
                if self.telegram_notifier:
                    try:
                        # Get approximate price from order or current price
                        price = float(order.get("avgPrice", 0))
                        if price == 0:
                            price = self.binance_client.get_current_price(symbol)

                        self.telegram_notifier.send_order_notification(
                            symbol=symbol,
                            side=side,
                            quantity=position_size,
                            price=price,
                            order_id=str(order.get("orderId")),
                            reduce_only=False,
                        )
                    except Exception as e:
                        logger.error(f"Error sending order notification: {e}")

            except Exception as e:
                logger.error(f"Error executing {side} order for {symbol}: {e}")
                results[symbol] = {
                    "status": "error",
                    "message": str(e),
                }

        # Phase 2: Check if all orders succeeded
        all_successful = all(r.get("status") == "success" for r in results.values())
        
        if not all_successful:
            logger.error(
                f"PARTIAL FILL DETECTED! Not all orders succeeded. "
                f"Successful: {len(successful_orders)}/{len(positions)}. "
                f"ROLLING BACK successful orders to prevent inconsistent position..."
            )
            
            # Rollback: Close all successful positions
            rollback_results = {}
            for order_info in successful_orders:
                try:
                    symbol = order_info["symbol"]
                    # Close by placing opposite side order
                    close_side = "SELL" if order_info["side"] == "BUY" else "BUY"
                    
                    logger.warning(
                        f"Rolling back: {close_side} {order_info['quantity']} {symbol} "
                        f"(original order_id={order_info['order_id']})"
                    )
                    
                    rollback_order = self.binance_client.place_market_order(
                        symbol, close_side, order_info["quantity"]
                    )
                    
                    rollback_results[symbol] = {
                        "status": "success",
                        "rollback_order_id": rollback_order.get("orderId"),
                    }
                    
                    logger.info(f"Rollback successful for {symbol}")
                    
                except Exception as e:
                    logger.error(
                        f"CRITICAL: Rollback failed for {symbol}! "
                        f"Manual intervention required. Error: {e}"
                    )
                    rollback_results[symbol] = {
                        "status": "error",
                        "message": str(e),
                    }
            
            return {
                "status": "error",
                "signal": signal,
                "action": "entry_failed_with_rollback",
                "orders": results,
                "rollback": rollback_results,
                "message": "Partial fill detected, all successful orders rolled back",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


        # RISK MANAGEMENT: Save trade entry time and capital to state
        total_capital = self.capital_per_leg * 2
        self._save_trade_entry(total_capital)

        return {
            "status": "success",
            "signal": signal,
            "action": "entry",
            "orders": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

                    # Send immediate notification
                    if self.telegram_notifier:
                        try:
                            # Get approximate price
                            price = float(order.get("avgPrice", 0))
                            if price == 0:
                                price = self.binance_client.get_current_price(symbol)

                            self.telegram_notifier.send_order_notification(
                                symbol=symbol,
                                side=order.get("side", "CLOSE"),
                                quantity=float(order.get("origQty", 0)),
                                price=price,
                                order_id=str(order.get("orderId")),
                                reduce_only=True,
                            )
                        except Exception as e:
                            logger.error(f"Error sending close notification: {e}")
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


        # RISK MANAGEMENT: Clear trade entry from state
        self._clear_trade_entry()

        return {
            "status": "success" if all(r.get("status") == "success" for r in results.values()) else "partial",
            "signal": "CLOSE",
            "action": "close",
            "orders": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
        
        IMPORTANT: Spread trading logic (per paper):
        - SHORT S1, LONG S2 → BUY ALT1, SELL ALT2 (creates LONG ALT1, SHORT ALT2)
        - LONG S1, SHORT S2 → SELL ALT1, BUY ALT2 (creates SHORT ALT1, LONG ALT2)
        
        This is because:
        - S1 = BTC - β1*ALT1. To SHORT S1 (decrease it), we need ALT1 to increase → BUY ALT1
        - S2 = BTC - β2*ALT2. To LONG S2 (increase it), we need ALT2 to decrease → SELL ALT2

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
        
        # DEBUG: Log what Binance actually returns
        logger.debug(
            f"Position detection: {self.copula_model.spread_pair.alt1}={alt1_amt}, "
            f"{self.copula_model.spread_pair.alt2}={alt2_amt} "
            f"(from Binance API)"
        )

        # Map actual positions to spread signal names
        # LONG ALT1 + SHORT ALT2 = SHORT S1, LONG S2 (by spread trading logic)
        if alt1_amt > 0 and alt2_amt < 0:
            return "SHORT_S1_LONG_S2"
        # SHORT ALT1 + LONG ALT2 = LONG S1, SHORT S2 (by spread trading logic)
        elif alt1_amt < 0 and alt2_amt > 0:
            return "LONG_S1_SHORT_S2"
        elif alt1_amt == 0 and alt2_amt == 0:
            return None
        else:
            logger.warning(
                f"Inconsistent positions: {self.copula_model.spread_pair.alt1}={alt1_amt}, "
                f"{self.copula_model.spread_pair.alt2}={alt2_amt}"
            )
            return "INCONSISTENT"

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

    # ========================================================================
    # RISK MANAGEMENT METHODS
    # ========================================================================

    def _get_position_pnl(self) -> float:
        """
        Get total unrealized PnL from Binance API.
        
        Returns:
            Total unrealized PnL across both legs in USDT
        """
        try:
            positions = self.binance_client.futures_position_information()
            
            alt1 = self.copula_model.spread_pair.alt1
            alt2 = self.copula_model.spread_pair.alt2
            
            pnl_alt1 = 0.0
            pnl_alt2 = 0.0
            
            for pos in positions:
                symbol = pos.get('symbol')
                if symbol == alt1:
                    pnl_alt1 = float(pos.get('unRealizedProfit', 0))
                elif symbol == alt2:
                    pnl_alt2 = float(pos.get('unRealizedProfit', 0))
            
            total_pnl = pnl_alt1 + pnl_alt2
            logger.debug(f"Position PnL: {alt1}=${pnl_alt1:.2f}, {alt2}=${pnl_alt2:.2f}, Total=${total_pnl:.2f}")
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error getting position PnL: {e}", exc_info=True)
            return 0.0

    def _get_trade_entry_time(self) -> Optional[datetime]:
        """
        Get trade entry time from state file.
        
        Returns:
            Entry time as datetime, or None if not found
        """
        if not self.state_manager:
            return None
        
        try:
            state = self.state_manager.load_state()
            entry_time_str = state.get('trade_entry_time')
            
            if entry_time_str:
                # Parse ISO format timestamp and ensure it's UTC-aware
                entry_dt = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                # Ensure timezone is set to UTC if naive
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                return entry_dt
            
        except Exception as e:
            logger.error(f"Error getting trade entry time: {e}", exc_info=True)
        
        return None

    def _save_trade_entry(self, capital: float):
        """
        Save trade entry time and capital to state file.
        
        Args:
            capital: Total position capital (both legs)
        """
        if not self.state_manager:
            logger.warning("No state_manager available, cannot save trade entry")
            return
        
        try:
            state = self.state_manager.load_state()
            state['trade_entry_time'] = datetime.now(timezone.utc).isoformat()
            state['trade_entry_capital'] = capital
            self.state_manager.save_state(state)
            
            logger.info(f"Saved trade entry: time={state['trade_entry_time']}, capital=${capital:.2f}")
            
        except Exception as e:
            logger.error(f"Error saving trade entry: {e}", exc_info=True)

    def _clear_trade_entry(self):
        """Clear trade entry data from state file."""
        if not self.state_manager:
            return
        
        try:
            state = self.state_manager.load_state()
            state.pop('trade_entry_time', None)
            state.pop('trade_entry_capital', None)
            self.state_manager.save_state(state)
            
            logger.info("Cleared trade entry from state")
            
        except Exception as e:
            logger.error(f"Error clearing trade entry: {e}", exc_info=True)
