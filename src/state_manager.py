"""State management for persisting trading configuration and results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .copula_model import SpreadPair
from .logger import get_logger

logger = get_logger(__name__)


class StateManager:
    """Manages persistence of trading state to disk."""

    def __init__(self, state_file: str = "state/state.json"):
        """
        Initialize state manager.

        Args:
            state_file: Path to state file
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save_formation_state(self, spread_pair: SpreadPair) -> None:
        """
        Save formation results to state file.

        Args:
            spread_pair: SpreadPair object with fitted parameters
        """
        try:
            state = {
                "formation_timestamp": datetime.now(datetime.UTC).isoformat(),
                "pair": {
                    "alt1": spread_pair.alt1,
                    "alt2": spread_pair.alt2,
                },
                "parameters": {
                    "beta1": float(spread_pair.beta1),
                    "beta2": float(spread_pair.beta2),
                    "rho": float(spread_pair.rho),
                    "tau": float(spread_pair.tau) if spread_pair.tau else None,
                },
                "spread_data": {
                    "spread1": spread_pair.spread1_data.tolist(),
                    "spread2": spread_pair.spread2_data.tolist(),
                },
                "current_position": None,
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"Formation state saved to {self.state_file}")

        except Exception as e:
            logger.error(f"Error saving formation state: {e}")
            raise

    def load_formation_state(self) -> Optional[SpreadPair]:
        """
        Load formation results from state file.

        Returns:
            SpreadPair object with fitted parameters, or None if no state exists
        """
        try:
            if not self.state_file.exists():
                logger.warning(f"State file not found: {self.state_file}")
                return None

            with open(self.state_file, "r") as f:
                state = json.load(f)

            # Reconstruct SpreadPair object
            pair = SpreadPair(
                alt1=state["pair"]["alt1"],
                alt2=state["pair"]["alt2"],
            )

            pair.beta1 = state["parameters"]["beta1"]
            pair.beta2 = state["parameters"]["beta2"]
            pair.rho = state["parameters"]["rho"]
            pair.tau = state["parameters"].get("tau")

            pair.spread1_data = np.array(state["spread_data"]["spread1"])
            pair.spread2_data = np.array(state["spread_data"]["spread2"])

            formation_time = state.get("formation_timestamp", "unknown")
            logger.info(
                f"Loaded formation state from {self.state_file} "
                f"(formed at {formation_time})"
            )

            return pair

        except Exception as e:
            logger.error(f"Error loading formation state: {e}")
            return None

    def update_position_state(self, position: Optional[str]) -> None:
        """
        Update current position state in state file.

        Args:
            position: Current position ('LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2', or None)
        """
        try:
            if not self.state_file.exists():
                logger.warning("State file doesn't exist, cannot update position")
                return

            with open(self.state_file, "r") as f:
                state = json.load(f)

            state["current_position"] = position
            state["position_updated_at"] = datetime.now(datetime.UTC).isoformat()

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

            logger.debug(f"Updated position state to: {position}")

        except Exception as e:
            logger.error(f"Error updating position state: {e}")

    def get_current_position(self) -> Optional[str]:
        """
        Get current position from state file.

        Returns:
            Current position string, or None if no position
        """
        try:
            if not self.state_file.exists():
                return None

            with open(self.state_file, "r") as f:
                state = json.load(f)

            return state.get("current_position")

        except Exception as e:
            logger.error(f"Error getting current position: {e}")
            return None

    def save_trade_log(self, trade_data: Dict) -> None:
        """
        Append trade execution to trade log file.

        Args:
            trade_data: Dict with trade execution details
        """
        try:
            log_file = self.state_file.parent / "trade_log.jsonl"

            with open(log_file, "a") as f:
                json.dump(trade_data, f)
                f.write("\n")

            logger.debug(f"Trade logged to {log_file}")

        except Exception as e:
            logger.error(f"Error saving trade log: {e}")

    def get_state_summary(self) -> Dict:
        """
        Get summary of current state.

        Returns:
            Dict with state summary
        """
        try:
            if not self.state_file.exists():
                return {"status": "no_state", "message": "No state file found"}

            with open(self.state_file, "r") as f:
                state = json.load(f)

            return {
                "status": "active",
                "formation_timestamp": state.get("formation_timestamp"),
                "pair": state.get("pair"),
                "current_position": state.get("current_position"),
                "position_updated_at": state.get("position_updated_at"),
            }

        except Exception as e:
            logger.error(f"Error getting state summary: {e}")
            return {"status": "error", "message": str(e)}

    def clear_state(self) -> None:
        """Clear state file (useful for testing or reset)."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                logger.info(f"Cleared state file: {self.state_file}")
        except Exception as e:
            logger.error(f"Error clearing state: {e}")
