"""Formation phase for selecting and fitting trading pairs."""

from datetime import datetime, timedelta, timezone
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
        volatility_jump_threshold: float = 0.30,
        volatility_match_factor: float = 1.3,
    ):
        """
        Initialize formation manager.

        Args:
            binance_client: BinanceClient instance
            altcoins: List of altcoin symbols (e.g., ['ETHUSDT', 'BNBUSDT'])
            formation_days: Number of days for formation period (default 21)
            interval: Kline interval (default '5m')
            volatility_jump_threshold: Max allowed single-day price jump (default 0.30 = 30%)
            volatility_match_factor: Max volatility ratio between pair legs (default 1.3)
        """
        self.binance_client = binance_client
        self.altcoins = altcoins
        self.formation_days = formation_days
        self.interval = interval
        self.volatility_jump_threshold = volatility_jump_threshold
        self.volatility_match_factor = volatility_match_factor
        self.btc_symbol = "BTCUSDT"
        self.blacklisted_coins = set()

    def run_formation(self, end_time: Optional[datetime] = None, output_dir: Optional[str] = None) -> Optional[SpreadPair]:
        """
        Run the formation phase to select and fit the best trading pair.

        Steps:
        1. Fetch historical data for BTC and all altcoins
        2. Calculate spreads for all pair combinations
        3. Test cointegration using ADF test
        4. Rank cointegrated pairs by Kendall's tau
        5. Select top pair
        6. Fit copula parameters

        Args:
            end_time: Optional end time for formation (used for backtesting). 
                      Defaults to datetime.now(timezone.utc).

        Returns:
            SpreadPair object with fitted parameters, or None if no suitable pairs found
        """
        logger.info("=" * 80)
        logger.info("FORMATION PHASE STARTED")
        logger.info("=" * 80)

        # Step 1: Fetch historical data
        if end_time is None:
            end_time = datetime.now(timezone.utc)
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

        # RISK MANAGEMENT: Filter out unstable coins and calculate volatilities
        self.blacklisted_coins = set()  # Reset blacklist each formation
        stable_altcoins = {}
        coin_volatilities = {}
        
        for symbol, df in altcoin_data.items():
            # Check coin stability (abnormal price jumps)
            if not self._check_coin_stability(symbol, df):
                continue  # Skip blacklisted coins
            
            # Calculate volatility
            volatility = self._calculate_volatility(df)
            stable_altcoins[symbol] = df
            coin_volatilities[symbol] = volatility
        
        logger.info(
            f"Coin stability check: {len(stable_altcoins)}/{len(altcoin_data)} passed. "
            f"Blacklisted: {self.blacklisted_coins if self.blacklisted_coins else 'None'}"
        )
        
        if len(stable_altcoins) < 2:
            logger.error(f"Insufficient stable coins: only {len(stable_altcoins)} available")
            return None

        # Step 2-4: Calculate spreads, test cointegration, rank by tau
        cointegrated_pairs: List[Tuple[SpreadPair, float]] = []

        # Generate all pair combinations from stable coins only
        alt_symbols = list(stable_altcoins.keys())
        for alt1, alt2 in combinations(alt_symbols, 2):
            try:
                # RISK MANAGEMENT: Check volatility match
                vol1 = coin_volatilities[alt1]
                vol2 = coin_volatilities[alt2]
                
                if not self._check_volatility_match(vol1, vol2):
                    logger.debug(
                        f"Skipping {alt1}-{alt2}: Volatility mismatch "
                        f"({vol1:.1%} vs {vol2:.1%}, ratio={max(vol1,vol2)/min(vol1,vol2):.2f})"
                    )
                    continue
                
                logger.info(f"\nAnalyzing pair: {alt1} - {alt2}")

                # Strict Alignment: Merge BTC, ALT1, and ALT2 on timestamp
                # We use inner join to keep only timestamps present in ALL three
                df_btc = btc_df[["timestamp", "close"]].rename(columns={"close": "btc"})
                df_alt1 = stable_altcoins[alt1][["timestamp", "close"]].rename(columns={"close": "alt1"})
                df_alt2 = stable_altcoins[alt2][["timestamp", "close"]].rename(columns={"close": "alt2"})

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

                # Calculate spread 1: S1 = BTC - β1*ALT1 (per paper)
                spread1, beta1 = calculate_spread(alt_prices=prices_alt1, btc_prices=prices_btc)
                pair.beta1 = beta1
                pair.spread1_data = spread1

                # Calculate spread 2: S2 = BTC - β2*ALT2 (per paper)
                spread2, beta2 = calculate_spread(alt_prices=prices_alt2, btc_prices=prices_btc)
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

        # Save candidates to CSV if output_dir is provided
        if output_dir:
            try:
                import os
                formations_dir = os.path.join(output_dir, "formations")
                os.makedirs(formations_dir, exist_ok=True)
                
                # Format timestamp for filename
                ts_str = end_time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(formations_dir, f"formation_candidates_{ts_str}.csv")
                
                candidates_data = []
                for pair, tau in cointegrated_pairs:
                    candidates_data.append({
                        "pair": f"{pair.alt1}-{pair.alt2}",
                        "tau": tau,
                        "beta1": pair.beta1,
                        "beta2": pair.beta2,
                        # We don't have p-value stored in pair object, but we logged it earlier
                        # For now, just saving what we have
                    })
                
                pd.DataFrame(candidates_data).to_csv(filename, index=False)
                logger.info(f"Saved {len(candidates_data)} formation candidates to {filename}")
            except Exception as e:
                logger.error(f"Error saving formation candidates CSV: {e}")

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs:")
        for i, (pair, tau) in enumerate(cointegrated_pairs[:5], 1):  # Show top 5
            logger.info(f"  {i}. {pair} - |τ|={tau:.4f}")

        best_pair, best_tau = cointegrated_pairs[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"SELECTED PAIR:")
        logger.info(f"  ALT1: {best_pair.alt1}")
        logger.info(f"  ALT2: {best_pair.alt2}")
        logger.info(f"  Kendall's |τ|: {best_tau:.4f}")
        logger.info(f"{'='*60}")

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
        logger.info(f"  ALT1({best_pair.alt1}): β1={best_pair.beta1:.6f}")
        logger.info(f"  ALT2({best_pair.alt2}): β2={best_pair.beta2:.6f}")
        logger.info(f"  Copula ρ={rho:.4f}, Kendall τ={best_pair.tau:.4f}")
        logger.info(f"  Spread formula: S = BTC - β*ALT (per paper)")
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

    def _check_coin_stability(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Check if coin has abnormal price jumps.
        
        Args:
            symbol: Coin symbol
            df: DataFrame with 'close' prices
            
        Returns:
            True if stable, False if should be blacklisted
        """
        # Resample to daily for jump detection
        df_daily = df.set_index('timestamp').resample('1D')['close'].last().dropna()
        
        if len(df_daily) < 2:
            return True  # Not enough data to check
        
        # Calculate daily returns
        returns = df_daily.pct_change().dropna()
        
        # Check for any single-day jump > threshold
        max_jump = returns.abs().max()
        
        if max_jump > self.volatility_jump_threshold:
            logger.warning(
                f"{symbol}: Abnormal price jump detected: {max_jump:.1%} > "
                f"{self.volatility_jump_threshold:.1%}. Blacklisting for this cycle."
            )
            self.blacklisted_coins.add(symbol)
            return False
        
        return True

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate annualized volatility from price data.
        
        Args:
            df: DataFrame with 'close' prices
            
        Returns:
            Annualized volatility
        """
        # Resample to daily for volatility calculation
        df_daily = df.set_index('timestamp').resample('1D')['close'].last().dropna()
        
        if len(df_daily) < 2:
            return 0.0
        
        returns = df_daily.pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualized

    def _check_volatility_match(self, vol1: float, vol2: float) -> bool:
        """
        Check if two volatilities are within acceptable ratio.
        
        Args:
            vol1: Volatility of first coin
            vol2: Volatility of second coin
            
        Returns:
            True if matched, False otherwise
        """
        if vol1 == 0 or vol2 == 0:
            return False
        
        ratio = max(vol1, vol2) / min(vol1, vol2)
        return ratio <= self.volatility_match_factor
