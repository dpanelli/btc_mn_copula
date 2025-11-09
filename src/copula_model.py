"""Copula-based trading model for spread pairs."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau
from statsmodels.tsa.stattools import adfuller

from .logger import get_logger

logger = get_logger(__name__)


class SpreadPair:
    """Represents a pair of spreads for copula trading."""

    def __init__(self, alt1: str, alt2: str):
        """
        Initialize spread pair.

        Args:
            alt1: First altcoin symbol (e.g., 'ETHUSDT')
            alt2: Second altcoin symbol (e.g., 'BNBUSDT')
        """
        self.alt1 = alt1
        self.alt2 = alt2
        self.beta1: Optional[float] = None
        self.beta2: Optional[float] = None
        self.spread1_data: Optional[np.ndarray] = None
        self.spread2_data: Optional[np.ndarray] = None
        self.rho: Optional[float] = None  # Gaussian copula correlation
        self.tau: Optional[float] = None  # Kendall's tau

    def __repr__(self) -> str:
        return f"SpreadPair({self.alt1}, {self.alt2})"


def calculate_spread(
    btc_prices: np.ndarray, alt_prices: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Calculate spread S = BTC - β*ALT using OLS regression.

    Args:
        btc_prices: BTC price series
        alt_prices: Altcoin price series

    Returns:
        Tuple of (spread series, beta coefficient)
    """
    # OLS regression: BTC = β*ALT + α
    # We use numpy's least squares
    X = alt_prices.reshape(-1, 1)
    y = btc_prices

    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # Solve for [α, β]
    coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    alpha, beta = coeffs

    # Calculate spread
    spread = btc_prices - beta * alt_prices

    logger.debug(f"OLS regression: β={beta:.6f}, α={alpha:.6f}")
    return spread, beta


def check_cointegration(spread: np.ndarray, significance_level: float = 0.05) -> bool:
    """
    Test if spread is stationary using Augmented Dickey-Fuller test.

    Args:
        spread: Spread series to test
        significance_level: Significance level for rejection (default 0.05)

    Returns:
        True if spread is stationary (cointegrated), False otherwise
    """
    try:
        # Perform ADF test
        result = adfuller(spread, autolag="AIC")
        adf_statistic, p_value = result[0], result[1]

        is_stationary = bool(p_value < significance_level)

        logger.debug(
            f"ADF test: statistic={adf_statistic:.4f}, p-value={p_value:.4f}, "
            f"stationary={is_stationary}"
        )
        return is_stationary

    except Exception as e:
        logger.error(f"Error in cointegration test: {e}")
        return False


def fit_empirical_cdf(data: np.ndarray) -> np.ndarray:
    """
    Transform data to uniform [0,1] using empirical CDF.

    Args:
        data: Data series to transform

    Returns:
        Uniform quantiles in [0,1]
    """
    # Rank data (ties averaged)
    ranks = stats.rankdata(data, method="average")

    # Transform to uniform: u = rank / (n + 1)
    # We use (n+1) to avoid u=1 which causes issues with inverse normal
    n = len(data)
    uniform = ranks / (n + 1)

    return uniform


def estimate_gaussian_copula_parameter(u1: np.ndarray, u2: np.ndarray) -> float:
    """
    Estimate Gaussian copula parameter ρ using method of moments.

    The relationship: τ = (2/π) * arcsin(ρ)
    Therefore: ρ = sin(π*τ/2)

    Args:
        u1: First uniform margin
        u2: Second uniform margin

    Returns:
        Correlation parameter ρ
    """
    # Calculate Kendall's tau
    tau, p_value = kendalltau(u1, u2)

    # Convert to Gaussian copula parameter
    rho = np.sin(np.pi * tau / 2)

    logger.debug(f"Kendall's tau={tau:.4f}, ρ={rho:.4f}")
    return rho


def gaussian_copula_conditional_cdf(
    u1: float, u2: float, rho: float, condition_on_2: bool = True
) -> float:
    """
    Calculate conditional CDF of Gaussian copula.

    h_1|2(u1|u2) = P(U1 ≤ u1 | U2 = u2) = Φ((Φ^-1(u1) - ρ*Φ^-1(u2)) / √(1-ρ^2))
    h_2|1(u2|u1) = P(U2 ≤ u2 | U1 = u1) = Φ((Φ^-1(u2) - ρ*Φ^-1(u1)) / √(1-ρ^2))

    Args:
        u1: First uniform quantile
        u2: Second uniform quantile
        rho: Gaussian copula correlation parameter
        condition_on_2: If True, compute h_1|2, else compute h_2|1

    Returns:
        Conditional probability
    """
    # Clip to avoid numerical issues at boundaries
    u1 = np.clip(u1, 1e-6, 1 - 1e-6)
    u2 = np.clip(u2, 1e-6, 1 - 1e-6)

    # Transform to standard normal quantiles
    z1 = stats.norm.ppf(u1)
    z2 = stats.norm.ppf(u2)

    # Calculate conditional CDF
    if condition_on_2:
        # h_1|2
        numerator = z1 - rho * z2
    else:
        # h_2|1
        numerator = z2 - rho * z1

    denominator = np.sqrt(1 - rho**2)
    conditional_prob = stats.norm.cdf(numerator / denominator)

    return conditional_prob


class CopulaModel:
    """Gaussian copula model for pairs trading."""

    def __init__(
        self,
        spread_pair: SpreadPair,
        entry_threshold: float = 0.10,
        exit_threshold: float = 0.10,
    ):
        """
        Initialize copula model.

        Args:
            spread_pair: SpreadPair object with fitted parameters
            entry_threshold: Entry threshold α1 (default 0.10)
            exit_threshold: Exit threshold α2 (default 0.10)
        """
        self.spread_pair = spread_pair
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def generate_signal(
        self, btc_price: float, alt1_price: float, alt2_price: float
    ) -> str:
        """
        Generate trading signal based on current prices.

        Args:
            btc_price: Current BTC price
            alt1_price: Current price of first altcoin
            alt2_price: Current price of second altcoin

        Returns:
            Signal: 'LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2', 'CLOSE', or 'HOLD'
        """
        # Calculate current spreads
        s1 = btc_price - self.spread_pair.beta1 * alt1_price
        s2 = btc_price - self.spread_pair.beta2 * alt2_price

        # Transform to uniform margins using historical data
        u1 = self._transform_to_uniform(s1, self.spread_pair.spread1_data)
        u2 = self._transform_to_uniform(s2, self.spread_pair.spread2_data)

        # Calculate conditional probabilities
        h_1_given_2 = gaussian_copula_conditional_cdf(
            u1, u2, self.spread_pair.rho, condition_on_2=True
        )
        h_2_given_1 = gaussian_copula_conditional_cdf(
            u1, u2, self.spread_pair.rho, condition_on_2=False
        )

        logger.debug(
            f"Spreads: S1={s1:.2f}, S2={s2:.2f} | "
            f"Uniforms: u1={u1:.4f}, u2={u2:.4f} | "
            f"Conditionals: h_1|2={h_1_given_2:.4f}, h_2|1={h_2_given_1:.4f}"
        )

        # Entry signals
        if (
            h_1_given_2 < self.entry_threshold
            and h_2_given_1 > 1 - self.entry_threshold
        ):
            logger.info(
                f"ENTRY SIGNAL: LONG S1, SHORT S2 "
                f"(h_1|2={h_1_given_2:.4f} < {self.entry_threshold}, "
                f"h_2|1={h_2_given_1:.4f} > {1-self.entry_threshold})"
            )
            return "LONG_S1_SHORT_S2"

        elif (
            h_1_given_2 > 1 - self.entry_threshold
            and h_2_given_1 < self.entry_threshold
        ):
            logger.info(
                f"ENTRY SIGNAL: SHORT S1, LONG S2 "
                f"(h_1|2={h_1_given_2:.4f} > {1-self.entry_threshold}, "
                f"h_2|1={h_2_given_1:.4f} < {self.entry_threshold})"
            )
            return "SHORT_S1_LONG_S2"

        # Exit signal (both near 0.5)
        elif (
            abs(h_1_given_2 - 0.5) < self.exit_threshold
            and abs(h_2_given_1 - 0.5) < self.exit_threshold
        ):
            logger.info(
                f"EXIT SIGNAL: CLOSE positions "
                f"(|h_1|2-0.5|={abs(h_1_given_2-0.5):.4f}, "
                f"|h_2|1-0.5|={abs(h_2_given_1-0.5):.4f} < {self.exit_threshold})"
            )
            return "CLOSE"

        return "HOLD"

    def _transform_to_uniform(
        self, value: float, historical_data: np.ndarray
    ) -> float:
        """
        Transform a value to uniform quantile using empirical CDF from historical data.

        Args:
            value: Current value to transform
            historical_data: Historical data used to build ECDF

        Returns:
            Uniform quantile in [0,1]
        """
        # Count how many historical values are <= current value
        rank = np.sum(historical_data <= value)
        n = len(historical_data)

        # Transform to uniform using same formula as fit_empirical_cdf
        uniform = (rank + 0.5) / (n + 1)  # +0.5 for continuity correction

        return np.clip(uniform, 1e-6, 1 - 1e-6)

    def get_position_quantities(
        self, signal: str, capital_per_leg: float
    ) -> Dict[str, Tuple[str, float]]:
        """
        Convert signal to actual position quantities.

        Signal translation:
        - LONG S1, SHORT S2 means:
          * S1 = BTC - β1*ALT1 increases → LONG β1*ALT1, keep BTC neutral
          * S2 = BTC - β2*ALT2 decreases → SHORT β2*ALT2, keep BTC neutral
          * Net: LONG β1*ALT1, SHORT β2*ALT2 (BTC cancels out as intermediary)

        Args:
            signal: Trading signal
            capital_per_leg: Capital to allocate per leg in USDT

        Returns:
            Dict mapping symbol to (side, capital_usdt)
        """
        if signal == "LONG_S1_SHORT_S2":
            # LONG spread 1 (LONG β1*ALT1), SHORT spread 2 (SHORT β2*ALT2)
            return {
                self.spread_pair.alt1: ("BUY", capital_per_leg),
                self.spread_pair.alt2: ("SELL", capital_per_leg),
            }
        elif signal == "SHORT_S1_LONG_S2":
            # SHORT spread 1 (SHORT β1*ALT1), LONG spread 2 (LONG β2*ALT2)
            return {
                self.spread_pair.alt1: ("SELL", capital_per_leg),
                self.spread_pair.alt2: ("BUY", capital_per_leg),
            }
        elif signal == "CLOSE":
            return {
                self.spread_pair.alt1: ("CLOSE", 0),
                self.spread_pair.alt2: ("CLOSE", 0),
            }
        else:  # HOLD
            return {}
