"""Formation phase for selecting and fitting trading pairs."""

from datetime import datetime, timedelta
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .binance_client import BinanceClient
from .copula_model import (
    SpreadPair,
    calculate_spread,
    check_cointegration,
    estimate_gaussian_copula_parameter,
    fit_empirical_cdf,
)
from .logger import get_logger

logger = get_logger(__name__)


class FormationManager:
    """Manages the weekly formation phase for pair selection."""

    def __init__(
        self,
        binance_client: BinanceClient,
        altcoins: List[str],
        formation_days: int = 21,
        interval: str = "5m",
    ):
        """
        Initialize formation manager.

        Args:
            binance_client: BinanceClient instance
            altcoins: List of altcoin symbols (e.g., ['ETHUSDT', 'BNBUSDT'])
            formation_days: Number of days for formation period (default 21)
            interval: Kline interval (default '5m')
        """
        self.binance_client = binance_client
        self.altcoins = altcoins
        self.formation_days = formation_days
        self.interval = interval
        self.btc_symbol = "BTCUSDT"

    def run_formation(self) -> Optional[SpreadPair]:
        """
        Run the formation phase to select and fit the best trading pair.

        Steps:
        1. Fetch historical data for BTC and all altcoins
        2. Calculate spreads for all pair combinations
        3. Test cointegration using ADF test
        4. Rank cointegrated pairs by Kendall's tau
        5. Select top pair
        6. Fit copula parameters

        Returns:
            SpreadPair object with fitted parameters, or None if no suitable pairs found
        """
        logger.info("=" * 80)
        logger.info("FORMATION PHASE STARTED")
        logger.info("=" * 80)

        # Step 1: Fetch historical data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=self.formation_days)

        logger.info(
            f"Fetching {self.formation_days} days of {self.interval} data "
            f"from {start_time} to {end_time}"
        )

        # Fetch BTC data
        btc_df = self.binance_client.get_historical_klines(
            self.btc_symbol, self.interval, start_time, end_time
        )
        btc_prices = btc_df["close"].values

        logger.info(f"Fetched {len(btc_prices)} BTC candles")

        # Fetch altcoin data
        altcoin_data: Dict[str, pd.DataFrame] = {}
        for alt in self.altcoins:
            try:
                df = self.binance_client.get_historical_klines(
                    alt, self.interval, start_time, end_time
                )
                if not df.empty:
                    altcoin_data[alt] = df
                    logger.info(f"Fetched {len(df)} {alt} candles")
            except Exception as e:
                logger.error(f"Error fetching {alt} data: {e}")

        if len(altcoin_data) < 2:
            logger.error(
                f"Insufficient altcoin data: only {len(altcoin_data)} coins available"
            )
            return None

        # Step 2-4: Calculate spreads, test cointegration, rank by tau
        cointegrated_pairs: List[Tuple[SpreadPair, float]] = []

        # Generate all pair combinations
        alt_symbols = list(altcoin_data.keys())
        for alt1, alt2 in combinations(alt_symbols, 2):
            try:
                logger.info(f"\nAnalyzing pair: {alt1} - {alt2}")

                # Strict Alignment: Merge BTC, ALT1, and ALT2 on timestamp
                # We use inner join to keep only timestamps present in ALL three
                df_btc = btc_df[["timestamp", "close"]].rename(columns={"close": "btc"})
                df_alt1 = altcoin_data[alt1][["timestamp", "close"]].rename(columns={"close": "alt1"})
                df_alt2 = altcoin_data[alt2][["timestamp", "close"]].rename(columns={"close": "alt2"})

                # Merge BTC and ALT1
                merged = pd.merge(df_btc, df_alt1, on="timestamp", how="inner")
                # Merge with ALT2
                merged = pd.merge(merged, df_alt2, on="timestamp", how="inner")

                if len(merged) < len(btc_df) * 0.9:  # Warn if we lose >10% of data
                    logger.warning(
                        f"  Data alignment dropped significant rows: "
                        f"{len(btc_df)} -> {len(merged)} candles"
                    )

                if len(merged) < 100:  # Minimum required data points
                    logger.warning("  Insufficient overlapping data, skipping pair")
                    continue

                # Extract aligned price arrays
                prices_btc = merged["btc"].values
                prices_alt1 = merged["alt1"].values
                prices_alt2 = merged["alt2"].values

                # Create spread pair
                pair = SpreadPair(alt1, alt2)

                # Calculate spread 1: BTC - β1*ALT1
                spread1, beta1 = calculate_spread(prices_btc, prices_alt1)
                pair.beta1 = beta1
                pair.spread1_data = spread1

                # Calculate spread 2: BTC - β2*ALT2
                spread2, beta2 = calculate_spread(prices_btc, prices_alt2)
                pair.beta2 = beta2
                pair.spread2_data = spread2

                # Test cointegration for both spreads
                is_coint_1 = check_cointegration(spread1)
                is_coint_2 = check_cointegration(spread2)

                logger.info(
                    f"  β1={beta1:.6f}, β2={beta2:.6f}, "
                    f"coint_S1={is_coint_1}, coint_S2={is_coint_2}"
                )

                if not (is_coint_1 and is_coint_2):
                    logger.info("  → Rejected: spreads not cointegrated")
                    continue

                # Calculate Kendall's tau between spreads
                tau, p_value = kendalltau(spread1, spread2)
                pair.tau = tau

                logger.info(
                    f"  → Cointegrated! Kendall's tau={tau:.4f} (p={p_value:.4f})"
                )

                cointegrated_pairs.append((pair, abs(tau)))

            except Exception as e:
                logger.error(f"Error analyzing pair {alt1}-{alt2}: {e}")

        if not cointegrated_pairs:
            logger.error("No cointegrated pairs found!")
            return None

        # Step 5: Select top pair by highest |tau|
        cointegrated_pairs.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs:")
        for i, (pair, tau) in enumerate(cointegrated_pairs[:5], 1):  # Show top 5
            logger.info(f"  {i}. {pair} - |τ|={tau:.4f}")

        best_pair, best_tau = cointegrated_pairs[0]
        logger.info(f"\nSelected pair: {best_pair} with |τ|={best_tau:.4f}")

        # Step 6: Fit copula parameters
        logger.info("\nFitting Gaussian copula...")

        # Transform spreads to uniform margins
        u1 = fit_empirical_cdf(best_pair.spread1_data)
        u2 = fit_empirical_cdf(best_pair.spread2_data)

        # Estimate copula parameter
        rho = estimate_gaussian_copula_parameter(u1, u2)
        best_pair.rho = rho

        logger.info(f"Copula fitted: ρ={rho:.4f}")

        logger.info("=" * 80)
        logger.info("FORMATION PHASE COMPLETED")
        logger.info("=" * 80)

        return best_pair

    def get_formation_summary(self, spread_pair: SpreadPair) -> Dict:
        """
        Get summary of formation results.

        Args:
            spread_pair: Fitted SpreadPair object

        Returns:
            Dict with formation summary
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": {
                "alt1": spread_pair.alt1,
                "alt2": spread_pair.alt2,
            },
            "parameters": {
                "beta1": float(spread_pair.beta1),
                "beta2": float(spread_pair.beta2),
                "rho": float(spread_pair.rho),
                "tau": float(spread_pair.tau),
            },
            "spread_data": {
                "spread1": spread_pair.spread1_data.tolist(),
                "spread2": spread_pair.spread2_data.tolist(),
            },
        }
