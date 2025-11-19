"""Telegram notification module for trading updates."""

import httpx
from typing import Dict, Optional

from .logger import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """Sends trading updates via Telegram bot."""

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot API token
            chat_id: Telegram chat ID to send messages to
            enabled: Whether notifications are enabled
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

        if self.enabled and self.bot_token and self.chat_id:
            logger.info(f"Telegram notifier initialized for chat_id: {self.chat_id}")
        else:
            logger.info("Telegram notifications disabled")

    def send_trading_update(
        self,
        positions: Dict,
        prices: Dict[str, float],
        signal: str,
        total_pnl: float,
        signal_data: Optional[Dict] = None,
    ) -> None:
        """
        Send a compact trading cycle update via Telegram.

        Args:
            positions: Dict of positions from get_current_positions()
            prices: Dict with keys 'btc', 'alt1', 'alt2' and their prices
            signal: Trading signal (HOLD, CLOSE, LONG_S1_SHORT_S2, SHORT_S1_LONG_S2)
            total_pnl: Total unrealized PnL
            signal_data: Optional dict with h_1_given_2, h_2_given_1, exit_threshold, distances
        """
        if not self.enabled:
            return

        try:
            message = self._format_trading_update(positions, prices, signal, total_pnl, signal_data)
            self._send_message(message)
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")

    def _send_message(self, message: str) -> None:
        """
        Send message via Telegram HTTP API (synchronous).

        Args:
            message: Message text to send
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    self.api_url,
                    json={
                        "chat_id": self.chat_id,
                        "text": message,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                )
                response.raise_for_status()
                logger.debug(f"Telegram message sent successfully")
        except httpx.HTTPError as e:
            logger.error(f"Telegram API error: {e}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    def _format_trading_update(
        self,
        positions: Dict,
        prices: Dict[str, float],
        signal: str,
        total_pnl: float,
        signal_data: Optional[Dict] = None,
    ) -> str:
        """
        Format trading update message with emojis.

        Args:
            positions: Position data
            prices: Price data
            signal: Trading signal
            total_pnl: Total P&L

        Returns:
            Formatted message string
        """
        lines = ["📊 <b>TRADING UPDATE</b>", ""]

        # Positions section
        has_positions = False
        position_lines = []

        for symbol, pos_data in positions.items():
            if "error" in pos_data:
                continue

            position_amt = pos_data.get("position_amt", 0)
            if position_amt == 0:
                continue

            has_positions = True
            entry_price = pos_data.get("entry_price", 0)
            unrealized_pnl = pos_data.get("unrealized_pnl", 0)
            side = "LONG" if position_amt > 0 else "SHORT"
            side_icon = "📈" if position_amt > 0 else "📉"
            notional = abs(position_amt) * entry_price

            # Format PnL with color
            pnl_icon = "💚" if unrealized_pnl >= 0 else "💔"
            pnl_str = f"{unrealized_pnl:+.2f}"

            position_lines.append(
                f"{symbol}: {side_icon} {side} | "
                f"{abs(position_amt):.4f} @ ${entry_price:.2f} | "
                f"${notional:.2f} | {pnl_icon} ${pnl_str}"
            )

        if has_positions:
            lines.append("💼 <b>Positions:</b>")
            lines.extend(position_lines)
            lines.append("━━━━━━━━━━━━━━━━━━━━")

            # Total PnL
            pnl_icon = "💚" if total_pnl >= 0 else "💔"
            lines.append(f"💵 <b>Total P&L:</b> {pnl_icon} ${total_pnl:+.2f}")
        else:
            lines.append("💼 <b>Positions:</b> FLAT (no open positions)")

        lines.append("")

        # Prices section
        btc_price = prices.get("btc", 0)
        alt1_price = prices.get("alt1", 0)
        alt2_price = prices.get("alt2", 0)
        alt1_symbol = prices.get("alt1_symbol", "ALT1")
        alt2_symbol = prices.get("alt2_symbol", "ALT2")

        lines.append(
            f"📍 <b>Prices:</b> BTC ${btc_price:,.2f} | "
            f"{alt1_symbol.replace('USDT', '')} ${alt1_price:.2f} | "
            f"{alt2_symbol.replace('USDT', '')} ${alt2_price:.2f}"
        )

        lines.append("")

        # Signal section (position-aware)
        signal_emoji = self._get_signal_emoji(signal)
        signal_text = self._get_signal_text(signal, has_positions)
        lines.append(f"🔔 <b>Signal:</b> {signal_emoji} {signal_text}")

        # Entry/Exit trigger information (if available)
        if signal_data:
            h_1_given_2 = signal_data.get("h_1_given_2")
            h_2_given_1 = signal_data.get("h_2_given_1")
            entry_threshold = signal_data.get("entry_threshold")
            exit_threshold = signal_data.get("exit_threshold")

            if h_1_given_2 is not None and h_2_given_1 is not None:
                lines.append("")

                if not has_positions:
                    # SHOW ENTRY TRIGGER INFO when FLAT
                    # LONG S1, SHORT S2: h_1|2 < α₁ AND h_2|1 > 1-α₁
                    long_distance_1 = abs(h_1_given_2 - entry_threshold) if h_1_given_2 > entry_threshold else 0
                    long_distance_2 = abs(h_2_given_1 - (1 - entry_threshold)) if h_2_given_1 < (1 - entry_threshold) else 0
                    long_distance = max(long_distance_1, long_distance_2)

                    # SHORT S1, LONG S2: h_1|2 > 1-α₁ AND h_2|1 < α₁
                    short_distance_1 = abs(h_1_given_2 - (1 - entry_threshold)) if h_1_given_2 < (1 - entry_threshold) else 0
                    short_distance_2 = abs(h_2_given_1 - entry_threshold) if h_2_given_1 > entry_threshold else 0
                    short_distance = max(short_distance_1, short_distance_2)

                    # Use the closest distance
                    min_distance = min(long_distance, short_distance)
                    closest_direction = "LONG S1" if long_distance <= short_distance else "SHORT S1"

                    # Status indicators
                    if min_distance < 0.05:
                        status_icon = "🟢"
                        status_text = "NEAR ENTRY"
                    elif min_distance < 0.10:
                        status_icon = "🟡"
                        status_text = "APPROACHING ENTRY"
                    else:
                        status_icon = "🔴"
                        status_text = "FAR FROM ENTRY"

                    lines.append(f"📈 <b>Entry Trigger:</b> {status_icon} {status_text}")
                    lines.append(f"• Threshold: {entry_threshold:.2f}")
                    lines.append(f"• h(1|2): {h_1_given_2:.4f}")
                    lines.append(f"  - LONG S1: needs &lt; {entry_threshold:.2f} (Δ: {long_distance_1:.4f})")
                    lines.append(f"  - SHORT S1: needs &gt; {1-entry_threshold:.2f} (Δ: {short_distance_1:.4f})")
                    lines.append(f"• h(2|1): {h_2_given_1:.4f}")
                    lines.append(f"  - LONG S1: needs &gt; {1-entry_threshold:.2f} (Δ: {long_distance_2:.4f})")
                    lines.append(f"  - SHORT S1: needs &lt; {entry_threshold:.2f} (Δ: {short_distance_2:.4f})")
                    lines.append(f"• Closest to {closest_direction}: {min_distance:.4f}")

                else:
                    # SHOW EXIT TRIGGER INFO when positions are open
                    distance_1 = abs(h_1_given_2 - 0.5)
                    distance_2 = abs(h_2_given_1 - 0.5)

                    # Determine how close we are to exit
                    max_distance = max(distance_1, distance_2)
                    if max_distance < exit_threshold:
                        status_icon = "🟢"
                        status_text = "NEAR EXIT"
                    elif max_distance < exit_threshold * 2:
                        status_icon = "🟡"
                        status_text = "APPROACHING EXIT"
                    else:
                        status_icon = "🔴"
                        status_text = "FAR FROM EXIT"

                    lines.append(f"📉 <b>Exit Trigger:</b> {status_icon} {status_text}")
                    lines.append(f"• Threshold: {exit_threshold:.2f}")
                    lines.append(f"• h(1|2): {h_1_given_2:.4f} (Δ from 0.5: {distance_1:.4f})")
                    lines.append(f"• h(2|1): {h_2_given_1:.4f} (Δ from 0.5: {distance_2:.4f})")

        return "\n".join(lines)

    def _get_signal_emoji(self, signal: str) -> str:
        """Get emoji for signal type."""
        if signal == "HOLD":
            return "⏸️"
        elif signal == "CLOSE":
            return "🔴"
        elif signal in ["LONG_S1_SHORT_S2", "SHORT_S1_LONG_S2"]:
            return "🟢"
        else:
            return "❓"

    def _get_signal_text(self, signal: str, has_positions: bool = True) -> str:
        """Get human-readable signal text (position-aware).
        
        Args:
            signal: Trading signal
            has_positions: Whether positions are currently open
            
        Returns:
            Human-readable signal text adjusted for position context
        """
        if signal == "HOLD":
            return "HOLD"
        elif signal == "CLOSE":
            # If FLAT, spreads are at equilibrium but nothing to close
            if not has_positions:
                return "HOLD (Spreads at equilibrium)"
            else:
                return "CLOSE POSITIONS"
        elif signal == "LONG_S1_SHORT_S2":
            return "LONG S1 / SHORT S2"
        elif signal == "SHORT_S1_LONG_S2":
            return "SHORT S1 / LONG S2"
        else:
            return signal

    def send_formation_update(self, pair_info: Dict) -> None:
        """
        Send formation phase completion notification.

        Args:
            pair_info: Dict with pair selection info (alt1, alt2, tau, rho, etc.)
        """
        if not self.enabled:
            return

        try:
            message = self._format_formation_update(pair_info)
            self._send_message(message)
        except Exception as e:
            logger.error(f"Error sending formation notification: {e}")

    def _format_formation_update(self, pair_info: Dict) -> str:
        """Format formation phase completion message."""
        alt1 = pair_info.get("alt1", "ALT1")
        alt2 = pair_info.get("alt2", "ALT2")
        tau = pair_info.get("tau", 0)
        rho = pair_info.get("rho", 0)
        beta1 = pair_info.get("beta1", 0)
        beta2 = pair_info.get("beta2", 0)

        lines = [
            "🎯 <b>FORMATION PHASE COMPLETE</b>",
            "",
            f"📌 <b>Selected Pair:</b> {alt1} - {alt2}",
            "",
            "<b>Parameters:</b>",
            f"• Kendall's τ: {tau:.4f}",
            f"• Copula ρ: {rho:.4f}",
            f"• β₁: {beta1:.6f}",
            f"• β₂: {beta2:.6f}",
            "",
            "✅ Ready to trade!",
        ]

        return "\n".join(lines)
