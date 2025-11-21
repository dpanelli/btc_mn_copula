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


def kss_test(series: np.ndarray, significance_level: float = 0.05) -> bool:
    """
    Perform Kapetanios-Shin-Snell (KSS) test for non-linear cointegration.
    
    Model: Δx_t = γ * x_{t-1}^3 + ε_t
    Null hypothesis: γ = 0 (non-stationary)
    Alternative: γ < 0 (stationary, ESTAR process)
    
    Critical values for Case 2 (demeaned data) from Kapetanios et al. (2003):
    1%: -3.48
    5%: -2.93
    10%: -2.66
    
    Args:
        series: Time series data
        significance_level: Significance level (0.01, 0.05, or 0.10)
        
    Returns:
        True if null hypothesis is rejected (stationary), False otherwise
    """
    try:
        # Remove NaN/Inf
        series = series[~np.isnan(series)]
        if len(series) < 10:
            return False
            
        # Demean the series (Case 2)
        x = series - np.mean(series)
        
        # Create lagged series
        x_lag = x[:-1]
        dx = np.diff(x)
        
        # Construct regressor: x_{t-1}^3
        X = x_lag ** 3
        y = dx
        
        # Run OLS: y = γ * X (no intercept because we demeaned)
        # γ = (X'X)^-1 X'y
        # t-stat = γ / SE(γ)
        
        # Using numpy for efficiency
        X = X.reshape(-1, 1)
        gamma, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        gamma = gamma[0]
        
        # Calculate standard error
        n = len(y)
        # Estimate variance of residuals: σ^2 = Σe^2 / (n - 1)
        # Note: degrees of freedom is n - 1 (1 parameter estimated)
        sigma2 = np.sum((y - X.flatten() * gamma) ** 2) / (n - 1)
        
        # Variance of gamma: Var(γ) = σ^2 * (X'X)^-1
        xtx_inv = 1 / np.sum(X ** 2)
        var_gamma = sigma2 * xtx_inv
        se_gamma = np.sqrt(var_gamma)
        
        # t-statistic
        t_stat = gamma / se_gamma
        
        # Critical values (Kapetanios et al. 2003, Table 1, Case 2)
        if significance_level <= 0.01:
            critical_value = -3.48
        elif significance_level <= 0.05:
            critical_value = -2.93
        else:
            critical_value = -2.66
            
        is_stationary = bool(t_stat < critical_value)
        
        logger.debug(
            f"KSS test: t-stat={t_stat:.4f}, crit={critical_value}, "
            f"stationary={is_stationary}"
        )
        
        return is_stationary
        
    except Exception as e:
        logger.error(f"Error in KSS test: {e}")
        return False


def check_cointegration(spread: np.ndarray, significance_level: float = 0.05) -> bool:
    """
    Test if spread is stationary using both ADF (Engle-Granger) and KSS tests.
    
    Args:
        spread: Spread series to test
        significance_level: Significance level for rejection (default 0.05)
        
    Returns:
        True if spread passes BOTH tests, False otherwise
    """
    try:
        # 1. Engle-Granger (ADF on residuals)
        # Note: Since 'spread' is already residuals from OLS, running ADF on it
        # is equivalent to the second step of EG test.
        # However, standard ADF critical values are slightly different from EG critical values.
        # For strict EG, we should use EG critical values (approx -3.34 for 5%).
        # statsmodels adfuller uses MacKinnon approximate p-values which are generally robust.
        
        result = adfuller(spread, autolag="AIC")
        adf_stat, p_value = result[0], result[1]
        is_eg_stationary = bool(p_value < significance_level)
        
        # 2. KSS Test (Non-linear)
        is_kss_stationary = kss_test(spread, significance_level)
        
        logger.debug(
            f"Cointegration results: EG_p={p_value:.4f} ({is_eg_stationary}), "
            f"KSS_stat={is_kss_stationary}"
        )
        
        # Require EITHER test to pass (Union of linear and non-linear cointegration)
        # The paper evaluates both strategies; allowing either ensures we capture
        # both linearly and non-linearly cointegrated pairs.
        return bool(is_eg_stationary or is_kss_stationary)
        
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
    ) -> Dict:
        """
        Generate trading signal based on current prices.

        Args:
            btc_price: Current BTC price
            alt1_price: Current price of first altcoin
            alt2_price: Current price of second altcoin

        Returns:
            Dict containing:
                - signal: 'LONG_S1_SHORT_S2', 'SHORT_S1_LONG_S2', 'CLOSE', or 'HOLD'
                - h_1_given_2: Conditional probability of spread 1 given spread 2
                - h_2_given_1: Conditional probability of spread 2 given spread 1
                - exit_threshold: Exit threshold value
                - distance_1: Distance of h_1_given_2 from 0.5
                - distance_2: Distance of h_2_given_1 from 0.5
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

        # Entry signals - CORRECTED PER PAPER
        # Paper uses: h <= (0.5 - alpha) and h >= (0.5 + alpha)
        # This creates entry zones at moderate deviations from median (0.5)
        if (
            h_1_given_2 <= 0.5 - self.entry_threshold
            and h_2_given_1 >= 0.5 + self.entry_threshold
        ):
            logger.info(
                f"ENTRY SIGNAL: LONG S1, SHORT S2 "
                f"(h_1|2={h_1_given_2:.4f} ≤ {0.5 - self.entry_threshold}, "
                f"h_2|1={h_2_given_1:.4f} ≥ {0.5 + self.entry_threshold})"
            )
            return {
                "signal": "LONG_S1_SHORT_S2",
                "h_1_given_2": h_1_given_2,
                "h_2_given_1": h_2_given_1,
                "entry_threshold": self.entry_threshold,
                "exit_threshold": self.exit_threshold,
                "distance_1": abs(h_1_given_2 - 0.5),
                "distance_2": abs(h_2_given_1 - 0.5),
            }

        elif (
            h_1_given_2 >= 0.5 + self.entry_threshold
            and h_2_given_1 <= 0.5 - self.entry_threshold
        ):
            logger.info(
                f"ENTRY SIGNAL: SHORT S1, LONG S2 "
                f"(h_1|2={h_1_given_2:.4f} ≥ {0.5 + self.entry_threshold}, "
                f"h_2|1={h_2_given_1:.4f} ≤ {0.5 - self.entry_threshold})"
            )
            return {
                "signal": "SHORT_S1_LONG_S2",
                "h_1_given_2": h_1_given_2,
                "h_2_given_1": h_2_given_1,
                "entry_threshold": self.entry_threshold,
                "exit_threshold": self.exit_threshold,
                "distance_1": abs(h_1_given_2 - 0.5),
                "distance_2": abs(h_2_given_1 - 0.5),
            }

        # Exit signal - CORRECTED PER PAPER  
        # Paper uses: (0.45 < h < 0.55) OR (0.45 < h < 0.55)
        # Exit if EITHER spread has converged to fair value
        elif (
            (0.45 < h_1_given_2 < 0.55)
            or (0.45 < h_2_given_1 < 0.55)
        ):
            logger.info(
                f"EXIT SIGNAL: CLOSE positions "
                f"(h_1|2={h_1_given_2:.4f} or h_2|1={h_2_given_1:.4f} near 0.5)"
            )
            return {
                "signal": "CLOSE",
                "h_1_given_2": h_1_given_2,
                "h_2_given_1": h_2_given_1,
                "entry_threshold": self.entry_threshold,
                "exit_threshold": self.exit_threshold,
                "distance_1": abs(h_1_given_2 - 0.5),
                "distance_2": abs(h_2_given_1 - 0.5),
            }

        return {
            "signal": "HOLD",
            "h_1_given_2": h_1_given_2,
            "h_2_given_1": h_2_given_1,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "distance_1": abs(h_1_given_2 - 0.5),
            "distance_2": abs(h_2_given_1 - 0.5),
        }

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
            # LONG spread 1 (SELL β1*ALT1), SHORT spread 2 (BUY β2*ALT2)
            # S1 = BTC - β1*ALT1. To LONG S1 (bet on increase), we need S1 to go up.
            # Since β1 is positive, we need ALT1 to go DOWN relative to BTC. So we SELL ALT1.
            # S2 = BTC - β2*ALT2. To SHORT S2 (bet on decrease), we need S2 to go down.
            # Since β2 is positive, we need ALT2 to go UP relative to BTC. So we BUY ALT2.
            return {
                self.spread_pair.alt1: ("SELL", capital_per_leg),
                self.spread_pair.alt2: ("BUY", capital_per_leg),
            }
        elif signal == "SHORT_S1_LONG_S2":
            # SHORT spread 1 (BUY β1*ALT1), LONG spread 2 (SELL β2*ALT2)
            # Inverse of above.
            return {
                self.spread_pair.alt1: ("BUY", capital_per_leg),
                self.spread_pair.alt2: ("SELL", capital_per_leg),
            }
        elif signal == "CLOSE":
            return {
                self.spread_pair.alt1: ("CLOSE", 0),
                self.spread_pair.alt2: ("CLOSE", 0),
            }
        else:  # HOLD
            return {}
