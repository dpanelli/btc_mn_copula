"""Trading execution manager for copula-based strategy."""

from datetime import datetime
from typing import Dict, Optional

from .binance_client import BinanceClient
from .copula_model import CopulaModel, SpreadPair
from .logger import get_logger

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
    ):
        """
        Initialize trading manager.

        Args:
            binance_client: BinanceClient instance
            capital_per_leg: Capital to allocate per leg in USDT
            max_leverage: Maximum leverage to use
            entry_threshold: Entry threshold α1 (default 0.10)
            exit_threshold: Exit threshold α2 (default 0.10)
        """
        self.binance_client = binance_client
        self.capital_per_leg = capital_per_leg
        self.max_leverage = max_leverage
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
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
                self.current_position = None
            elif signal in ["LONG_S1_SHORT_S2", "SHORT_S1_LONG_S2"]:
                # Don't enter new position if already in one
                if self.current_position is not None and self.current_position != signal:
                    logger.info(
                        f"Already in position {self.current_position}, closing before entering new one"
                    )
                    self._close_positions()

                result = self._execute_entry_signal(signal)
                if result["status"] == "success":
                    self.current_position = signal
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
