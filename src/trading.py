"""Trading execution manager for copula-based strategy."""

from datetime import datetime, timezone
from typing import Dict, Optional

from .binance_client import BinanceClient
from .copula_model import CopulaModel, SpreadPair
from .strategy import PairsTradingStrategy
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
        take_profit_pct: float = 0.10,
        max_trade_duration_hours: int = 48,
        cooldown_minutes: int = 60,
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
            take_profit_pct: Take-profit as % of position value (default 0.10 = 10%)
            max_trade_duration_hours: Maximum trade duration in hours (default 48)
            cooldown_minutes: Cooldown period in minutes after forced exit (default 60)
        """
        self.binance_client = binance_client
        self.capital_per_leg = capital_per_leg
        self.max_leverage = max_leverage
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.telegram_notifier = telegram_notifier
        self.state_manager = state_manager
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trade_duration_hours = max_trade_duration_hours
        self.cooldown_minutes = cooldown_minutes
        self.strategy: Optional[PairsTradingStrategy] = None
        self.current_position: Optional[str] = None  # Track current position state
        self.btc_symbol = "BTCUSDT"

    def set_spread_pair(self, spread_pair: SpreadPair) -> None:
        """
        Set the spread pair to trade and initialize copula model.

        Args:
            spread_pair: SpreadPair object with fitted parameters
        """
        self.strategy = PairsTradingStrategy(
            spread_pair=spread_pair,
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            capital_per_leg=self.capital_per_leg,
            max_leverage=self.max_leverage
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
        if self.strategy is None:
            logger.info("No strategy set - waiting for formation phase to complete")
            return {"status": "waiting", "message": "No strategy - waiting for formation"}

        logger.info("-" * 80)
        logger.info(f"TRADING CYCLE: {datetime.now(timezone.utc).isoformat()}")
        logger.info("-" * 80)

        # NO-TRADE ZONE: Check if we are within 15 minutes of formation
        if self.state_manager:
            try:
                state = self.state_manager.load_state()
                formation_time_str = state.get('formation_timestamp')
                if formation_time_str:
                    formation_dt = datetime.fromisoformat(formation_time_str.replace('Z', '+00:00'))
                    if formation_dt.tzinfo is None:
                        formation_dt = formation_dt.replace(tzinfo=timezone.utc)
                    
                    current_time = datetime.now(timezone.utc)
                    time_since_formation = (current_time - formation_dt).total_seconds() / 60
                    
                    if time_since_formation < 15:
                        logger.info(f"In No-Trade Zone ({time_since_formation:.1f}m since formation). Waiting...")
                        return {
                            "status": "waiting", 
                            "message": f"No-Trade Zone: {time_since_formation:.1f}m < 15m"
                        }
            except Exception as e:
                logger.error(f"Error checking formation time: {e}")

        # COOLDOWN CHECK: Check if we are in a cooldown period
        if self.state_manager:
            try:
                state = self.state_manager.load_state()
                cooldown_until_str = state.get('cooldown_until')
                if cooldown_until_str:
                    cooldown_dt = datetime.fromisoformat(cooldown_until_str.replace('Z', '+00:00'))
                    if cooldown_dt.tzinfo is None:
                        cooldown_dt = cooldown_dt.replace(tzinfo=timezone.utc)
                    
                    current_time = datetime.now(timezone.utc)
                    if current_time < cooldown_dt:
                        remaining_minutes = (cooldown_dt - current_time).total_seconds() / 60
                        logger.info(f"❄️ COOLDOWN ACTIVE: Trading paused for {remaining_minutes:.1f} more minutes")
                        return {
                            "status": "waiting",
                            "message": f"Cooldown active: {remaining_minutes:.1f}m remaining"
                        }
                    else:
                        # Cooldown expired, clear it
                        logger.info("Cooldown expired, resuming trading")
                        self.state_manager.update_cooldown(None)
            except Exception as e:
                logger.error(f"Error checking cooldown: {e}")

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
                
                # Activate cooldown
                self._activate_cooldown()
                
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

            # 2. Take-profit check (percentage-based)
            take_profit_threshold = entry_capital * self.take_profit_pct

            if unrealized_pnl >= take_profit_threshold:
                logger.info(
                    f"💰 TAKE-PROFIT triggered: PnL=${unrealized_pnl:.2f} >= "
                    f"${take_profit_threshold:.2f} ({self.take_profit_pct:.1%} of ${entry_capital:.2f})"
                )
                self._close_positions()
                self._clear_trade_entry()

                # No cooldown for take-profit - allow immediate re-entry

                if self.telegram_notifier:
                    self.telegram_notifier.send_message(
                        f"💰 TAKE-PROFIT: Closed positions at ${unrealized_pnl:.2f} "
                        f"(+{self.take_profit_pct:.0%} target hit)"
                    )

                return {
                    "status": "take_profit",
                    "pnl": unrealized_pnl,
                    "threshold": take_profit_threshold,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # 3. Time-based exit check (skip if disabled with -1)
            entry_time = self._get_trade_entry_time()
            if entry_time and self.max_trade_duration_hours > 0:
                duration_hours = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

                if duration_hours > self.max_trade_duration_hours:
                    logger.warning(
                        f"⏰ TIME-BASED EXIT: Trade duration {duration_hours:.1f}h > "
                        f"{self.max_trade_duration_hours}h"
                    )
                    self._close_positions()
                    self._clear_trade_entry()
                    
                    # Activate cooldown
                    self._activate_cooldown()
                    
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
                self.strategy.spread_pair.alt1
            )
            alt2_price = self.binance_client.get_current_price(
                self.strategy.spread_pair.alt2
            )

            logger.info(
                f"Current prices: BTC={btc_price:.2f}, "
                f"{self.strategy.spread_pair.alt1}={alt1_price:.4f}, "
                f"{self.strategy.spread_pair.alt2}={alt2_price:.4f}"
            )

            # Check if spreads are out of historical range (stale formation)
            out_of_range_result = self._check_spread_out_of_range(
                btc_price, alt1_price, alt2_price
            )
            if out_of_range_result:
                return out_of_range_result

            # Generate signal
            signal_obj = self.strategy.generate_signal(btc_price, alt1_price, alt2_price)
            signal = signal_obj.signal
            signal_data = signal_obj.metadata

            logger.info(f"Signal: {signal}")

            # Initialize result variable
            result = None

            # Check if we need to act on the signal
            if signal == "HOLD":
                logger.info("Signal is HOLD - no action taken")
                result = {
                    "status": "success",
                    "signal": signal,
                    "action": "none",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            # Execute trades based on signal
            elif signal == "CLOSE":
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
                        result = {
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
                            logger.error("Failed to close positions before entering new trade - ABORTING ENTRY")
                            result = {
                                "status": "error",
                                "message": "Failed to close existing positions",
                                "signal": signal
                            }

                # Only execute entry if no position exists (and not already set result above)
                if result is None:
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
                result = {
                    "status": "error",
                    "message": f"Unknown signal: {signal}",
                }

            # Send Telegram notification AFTER trade execution
            # This ensures the notification reflects the actual state after signal processing
            if self.telegram_notifier:
                try:
                    positions = self.get_current_positions()

                    # Calculate total PnL
                    total_pnl = sum(
                        pos.get("unrealized_pnl", 0)
                        for pos in positions.values()
                        if "error" not in pos
                    )

                    price_data = {
                        "btc": btc_price,
                        "alt1": alt1_price,
                        "alt2": alt2_price,
                        "alt1_symbol": self.strategy.spread_pair.alt1,
                        "alt2_symbol": self.strategy.spread_pair.alt2,
                    }

                    # Use "EXECUTED" signal name if entry was successful, otherwise show original signal
                    display_signal = signal
                    if result and result.get("status") == "error":
                        display_signal = f"{signal} (FAILED)"

                    self.telegram_notifier.send_trading_update(
                        positions=positions,
                        prices=price_data,
                        signal=display_signal,
                        total_pnl=total_pnl,
                        signal_data=signal_data,
                    )
                except Exception as e:
                    logger.error(f"Error sending Telegram notification: {e}")

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
        # Need current prices for sizing
        prices = {
            self.strategy.spread_pair.alt1: self.binance_client.get_current_price(self.strategy.spread_pair.alt1),
            self.strategy.spread_pair.alt2: self.binance_client.get_current_price(self.strategy.spread_pair.alt2)
        }
        
        positions = self.strategy.get_target_positions(signal, prices)

        # Check if we have enough balance for all legs
        # Required margin = (2 * capital_per_leg) / leverage
        # positions contains (side, quantity), not dollar amounts
        total_notional = 2 * self.capital_per_leg  # $100 per leg × 2 legs = $200
        required_margin = total_notional / self.max_leverage  # $200 / 4 = $50
        try:
            available_balance = self.binance_client.get_account_balance()
            if available_balance < required_margin:
                logger.error(
                    f"Insufficient balance for entry: Available=${available_balance:.2f}, "
                    f"Required margin=${required_margin:.2f} (notional=${total_notional:.2f}, leverage={self.max_leverage}x)"
                )
                return {
                    "status": "error",
                    "message": "Insufficient balance",
                    "available": available_balance,
                    "required": required_margin,
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
        for symbol, (side, quantity) in positions.items():
            try:
                # Calculate position size (quantity is already in base asset units from strategy)
                # But binance_client.calculate_position_size expects capital if we were using old logic.
                # Wait, strategy.get_target_positions returns (side, quantity_asset).
                # Let's double check strategy.py.
                # Yes: return {symbol: ("BUY", capital / price)}
                # So 'quantity' here is the actual amount of asset to buy/sell.
                
                # We should verify precision/min qty using binance client helper if available,
                # or just pass it to place_market_order which usually handles precision if passed as string,
                # but here we might need to round it.
                # Let's use calculate_position_size but we need to reverse engineer capital?
                # No, let's just use the quantity directly but ensure precision.
                # Actually, existing code used calculate_position_size(symbol, capital, leverage).
                # Strategy now gives us quantity.
                
                # Let's trust the strategy's math but maybe round it?
                # Ideally BinanceClient should have a method to normalize quantity.
                # For now, let's assume the quantity is raw float and we need to format it.
                
                # REVISION: To minimize changes and risk, let's convert back to capital for the existing method
                # OR better, update this loop to use quantity directly.
                
                # Round quantity to exchange precision (LOT_SIZE step size)
                # This is critical - Binance will reject orders with invalid precision
                position_size = self.binance_client.round_quantity_to_precision(symbol, quantity)
                
                # Place order with properly rounded quantity
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
        if self.strategy is None:
            return {"status": "error", "message": "No strategy"}

        symbols = [
            self.strategy.spread_pair.alt1,
            self.strategy.spread_pair.alt2,
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


        # Check if all closes were successful
        all_success = all(r.get("status") == "success" for r in results.values())
        
        if all_success:
            # RISK MANAGEMENT: Clear trade entry from state only if fully closed
            self._clear_trade_entry()
        else:
            logger.error("Failed to close all positions! Keeping trade entry in state.")

        return {
            "status": "success" if all_success else "partial",
            "signal": "CLOSE",
            "action": "close",
            "orders": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return positions

    def get_current_positions(self, max_retries: int = 3) -> Dict:
        """
        Get current position information for both symbols with retry logic.

        Args:
            max_retries: Number of retries for API calls

        Returns:
            Dict with position info for each symbol
        """
        if self.strategy is None:
            return {}

        positions = {}
        import time
        
        for symbol in [
            self.strategy.spread_pair.alt1,
            self.strategy.spread_pair.alt2,
        ]:
            for attempt in range(max_retries):
                try:
                    pos = self.binance_client.get_position(symbol)
                    positions[symbol] = pos if pos else {"position_amt": 0}
                    break # Success
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Error fetching position for {symbol} after {max_retries} attempts: {e}")
                        positions[symbol] = {"error": str(e)}
                    else:
                        logger.warning(f"Error fetching position for {symbol} (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                        time.sleep(1) # Wait 1s before retry

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
        
        FAIL-SAFE: Returns True if API error occurs to prevent double-entry.

        Returns:
            True if any position exists OR if API error occurs, False otherwise
        """
        if self.strategy is None:
            return False

        positions = self.get_current_positions()
        for symbol, pos_data in positions.items():
            # If there's an API error, assume we might have a position (Fail Safe)
            if "error" in pos_data:
                logger.warning(f"API error for {symbol} in _has_open_positions, assuming True for safety")
                return True
                
            if pos_data.get("position_amt", 0) != 0:
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
            'LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2', 'INCONSISTENT', or None
            Returns None if API error occurs (to prevent false INCONSISTENT)
        """
        if self.strategy is None:
            return None

        positions = self.get_current_positions()
        
        # Check for API errors first
        for symbol, pos_data in positions.items():
            if "error" in pos_data:
                logger.error(f"Cannot determine position type due to API error for {symbol}")
                return None # Return None to indicate unknown state (safe fallback)
        
        # Convert Binance positions to simple dict {symbol: signed_quantity} for strategy
        strategy_positions = {}
        for symbol, pos_data in positions.items():
            amt = float(pos_data.get("position_amt", 0))
            strategy_positions[symbol] = amt
                
        return self.strategy.get_position_state(strategy_positions)

    def _log_positions_and_pnl(self) -> None:
        """
        Log detailed position and PnL information from Binance.
        Logs each position (symbol, size, side, PnL) and total account PnL.
        """
        if self.strategy is None:
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
            for symbol in [self.strategy.spread_pair.alt1, self.strategy.spread_pair.alt2]:
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

    def _activate_cooldown(self) -> None:
        """Activate cooldown period after forced exit."""
        if not self.state_manager:
            return
            
        from datetime import timedelta
        cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.cooldown_minutes)
        self.state_manager.update_cooldown(cooldown_until)
        logger.info(f"❄️ Cooldown activated until {cooldown_until.isoformat()} ({self.cooldown_minutes} mins)")

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
            
            alt1 = self.strategy.spread_pair.alt1
            alt2 = self.strategy.spread_pair.alt2
            
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
            if state is None:
                state = {}
                
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
            if state:
                state.pop('trade_entry_time', None)
                state.pop('trade_entry_capital', None)
                self.state_manager.save_state(state)
                logger.info("Cleared trade entry from state")

        except Exception as e:
            logger.error(f"Error clearing trade entry: {e}", exc_info=True)

    def _check_spread_out_of_range(
        self, btc_price: float, alt1_price: float, alt2_price: float
    ) -> Optional[Dict]:
        """
        Check if current spreads are outside the historical formation range.

        If spreads are out of range, the copula calibration is stale and formation
        needs to be re-run. This closes any open positions and signals for re-formation.

        Args:
            btc_price: Current BTC price
            alt1_price: Current ALT1 price
            alt2_price: Current ALT2 price

        Returns:
            Dict with status="out_of_range" if spreads are outside historical range,
            None otherwise (continue normal processing)
        """
        if self.strategy is None:
            return None

        spread_pair = self.strategy.spread_pair

        # Calculate current spreads
        s1_current = btc_price - spread_pair.beta1 * alt1_price
        s2_current = btc_price - spread_pair.beta2 * alt2_price

        # Get historical range
        s1_min, s1_max = spread_pair.spread1_data.min(), spread_pair.spread1_data.max()
        s2_min, s2_max = spread_pair.spread2_data.min(), spread_pair.spread2_data.max()

        # Check if either spread is outside historical range
        s1_out = s1_current < s1_min or s1_current > s1_max
        s2_out = s2_current < s2_min or s2_current > s2_max

        if not (s1_out or s2_out):
            return None  # Spreads are in range, continue normal processing

        # Only trigger out-of-range if we have open positions to protect
        # If flat, allow trading to continue - the copula will just generate extreme signals
        if not self._has_open_positions():
            logger.info(
                f"Spread out of range but no positions open - continuing normal trading"
            )
            return None

        # Log which spread(s) are out of range
        out_of_range_details = []
        if s1_out:
            out_of_range_details.append(
                f"S1={s1_current:.2f} outside [{s1_min:.2f}, {s1_max:.2f}]"
            )
        if s2_out:
            out_of_range_details.append(
                f"S2={s2_current:.2f} outside [{s2_min:.2f}, {s2_max:.2f}]"
            )

        logger.warning(
            f"📊 SPREAD OUT OF RANGE: {', '.join(out_of_range_details)}. "
            f"Copula calibration is stale - need to re-run formation."
        )

        # Close any open positions
        unrealized_pnl = 0.0
        if self._has_open_positions():
            unrealized_pnl = self._get_position_pnl()
            logger.info(f"Closing positions before formation re-run (PnL=${unrealized_pnl:.2f})")
            self._close_positions()
            self._clear_trade_entry()
            self.current_position = None

        # Send Telegram notification
        if self.telegram_notifier:
            self.telegram_notifier.send_message(
                f"📊 SPREAD OUT OF RANGE: {', '.join(out_of_range_details)}\n"
                f"Closing positions (PnL=${unrealized_pnl:.2f}) and re-running formation."
            )

        return {
            "status": "out_of_range",
            "action": "trigger_formation",
            "s1_current": s1_current,
            "s2_current": s2_current,
            "s1_range": [s1_min, s1_max],
            "s2_range": [s2_min, s2_max],
            "pnl": unrealized_pnl,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
