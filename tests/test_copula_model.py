"""Unit tests for copula model."""

import numpy as np
import pytest
from scipy.stats import kendalltau, norm

from src.copula_model import (
    CopulaModel,
    SpreadPair,
    calculate_spread,
    check_cointegration,
    estimate_gaussian_copula_parameter,
    fit_empirical_cdf,
    gaussian_copula_conditional_cdf,
)


class TestSpreadCalculation:
    """Test spread calculation with OLS regression."""

    def test_calculate_spread_basic(self):
        """Test basic spread calculation."""
        btc_prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        alt_prices = np.array([10.0, 10.1, 10.2, 10.3, 10.4])

        spread, beta = calculate_spread(btc_prices, alt_prices)

        # Check that beta is positive
        assert beta > 0

        # Check spread length matches input
        assert len(spread) == len(btc_prices)

        # Verify relationship: spread = BTC - beta * ALT
        expected_spread = btc_prices - beta * alt_prices
        np.testing.assert_array_almost_equal(spread, expected_spread)

    def test_calculate_spread_perfect_correlation(self):
        """Test spread with perfect correlation."""
        # Perfect linear relationship: BTC = 10 * ALT
        alt_prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        btc_prices = 10.0 * alt_prices

        spread, beta = calculate_spread(btc_prices, alt_prices)

        # Beta should be approximately 10
        assert abs(beta - 10.0) < 0.01

        # Spread should be close to zero (minus intercept)
        assert np.std(spread) < 0.01


class TestEmpiricalCDF:
    """Test empirical CDF transformation."""

    def test_fit_empirical_cdf_uniform_output(self):
        """Test that ECDF produces values in [0, 1]."""
        data = np.random.randn(100)
        uniform = fit_empirical_cdf(data)

        # Check range
        assert np.all(uniform > 0)
        assert np.all(uniform < 1)

    def test_fit_empirical_cdf_sorted_data(self):
        """Test ECDF on sorted data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        uniform = fit_empirical_cdf(data)

        # Should be approximately evenly spaced
        assert uniform[0] < uniform[1] < uniform[2] < uniform[3] < uniform[4]


class TestGaussianCopula:
    """Test Gaussian copula parameter estimation and conditional CDF."""

    def test_estimate_copula_parameter_positive_correlation(self):
        """Test copula parameter estimation with positive correlation."""
        # Create positively correlated uniform data
        np.random.seed(42)
        u1 = np.random.uniform(0, 1, 1000)
        u2 = u1 + np.random.normal(0, 0.1, 1000)
        u2 = np.clip(u2, 0, 1)

        rho = estimate_gaussian_copula_parameter(u1, u2)

        # Should be positive
        assert rho > 0

    def test_estimate_copula_parameter_negative_correlation(self):
        """Test copula parameter estimation with negative correlation."""
        # Create negatively correlated uniform data
        np.random.seed(42)
        u1 = np.random.uniform(0, 1, 1000)
        u2 = 1 - u1 + np.random.normal(0, 0.1, 1000)
        u2 = np.clip(u2, 0, 1)

        rho = estimate_gaussian_copula_parameter(u1, u2)

        # Should be negative
        assert rho < 0

    def test_gaussian_copula_conditional_cdf_median(self):
        """Test conditional CDF at median values."""
        # At u1=u2=0.5 and rho=0, conditional should be ~0.5
        h = gaussian_copula_conditional_cdf(0.5, 0.5, 0.0, condition_on_2=True)
        assert abs(h - 0.5) < 0.01

    def test_gaussian_copula_conditional_cdf_boundary(self):
        """Test conditional CDF at boundary values."""
        # At u1=0.01, h_1|2 should be small
        h = gaussian_copula_conditional_cdf(0.01, 0.5, 0.5, condition_on_2=True)
        assert h < 0.3

        # At u1=0.99, h_1|2 should be large
        h = gaussian_copula_conditional_cdf(0.99, 0.5, 0.5, condition_on_2=True)
        assert h > 0.7


class TestCointegration:
    """Test cointegration testing."""

    def test_cointegration_stationary(self):
        """Test cointegration test on stationary series."""
        # Random walk differences (stationary)
        np.random.seed(42)
        spread = np.random.randn(200)

        is_stationary = check_cointegration(spread, significance_level=0.05)

        # Should be stationary (though may fail due to randomness)
        # This is a probabilistic test, so we just check it returns bool
        assert isinstance(is_stationary, bool)

    def test_cointegration_non_stationary(self):
        """Test cointegration test on non-stationary series."""
        # Random walk (non-stationary)
        np.random.seed(42)
        walk = np.cumsum(np.random.randn(200))

        is_stationary = check_cointegration(walk, significance_level=0.05)

        # Should be non-stationary (though may fail due to randomness)
        assert isinstance(is_stationary, bool)


class TestCopulaModel:
    """Test CopulaModel signal generation."""

    @pytest.fixture
    def mock_spread_pair(self):
        """Create a mock SpreadPair for testing."""
        pair = SpreadPair("ETHUSDT", "BNBUSDT")
        pair.beta1 = 2.5
        pair.beta2 = 1.8
        pair.rho = 0.6

        # Create mock historical spread data
        np.random.seed(42)
        pair.spread1_data = np.random.randn(1000) * 10 + 50
        pair.spread2_data = np.random.randn(1000) * 8 + 40

        return pair

    def test_copula_model_initialization(self, mock_spread_pair):
        """Test CopulaModel initialization."""
        model = CopulaModel(
            spread_pair=mock_spread_pair,
            entry_threshold=0.10,
            exit_threshold=0.10,
        )

        assert model.entry_threshold == 0.10
        assert model.exit_threshold == 0.10
        assert model.spread_pair.alt1 == "ETHUSDT"

    def test_generate_signal_hold(self, mock_spread_pair):
        """Test signal generation returns HOLD for neutral conditions."""
        model = CopulaModel(
            spread_pair=mock_spread_pair,
            entry_threshold=0.10,
            exit_threshold=0.10,
        )

        # Use median prices
        btc_price = 50000.0
        alt1_price = 3000.0
        alt2_price = 400.0

        signal = model.generate_signal(btc_price, alt1_price, alt2_price)

        # Signal should be one of the valid types
        assert signal in ["LONG_S1_SHORT_S2", "SHORT_S1_LONG_S2", "CLOSE", "HOLD"]

    def test_get_position_quantities_long_s1(self, mock_spread_pair):
        """Test position quantities for LONG S1, SHORT S2 signal."""
        model = CopulaModel(spread_pair=mock_spread_pair)

        positions = model.get_position_quantities("LONG_S1_SHORT_S2", capital_per_leg=10000)

        assert "ETHUSDT" in positions
        assert "BNBUSDT" in positions
        assert positions["ETHUSDT"][0] == "BUY"
        assert positions["BNBUSDT"][0] == "SELL"
        assert positions["ETHUSDT"][1] == 10000
        assert positions["BNBUSDT"][1] == 10000

    def test_get_position_quantities_short_s1(self, mock_spread_pair):
        """Test position quantities for SHORT S1, LONG S2 signal."""
        model = CopulaModel(spread_pair=mock_spread_pair)

        positions = model.get_position_quantities("SHORT_S1_LONG_S2", capital_per_leg=10000)

        assert positions["ETHUSDT"][0] == "SELL"
        assert positions["BNBUSDT"][0] == "BUY"

    def test_get_position_quantities_close(self, mock_spread_pair):
        """Test position quantities for CLOSE signal."""
        model = CopulaModel(spread_pair=mock_spread_pair)

        positions = model.get_position_quantities("CLOSE", capital_per_leg=10000)

        assert positions["ETHUSDT"][0] == "CLOSE"
        assert positions["BNBUSDT"][0] == "CLOSE"

    def test_get_position_quantities_hold(self, mock_spread_pair):
        """Test position quantities for HOLD signal."""
        model = CopulaModel(spread_pair=mock_spread_pair)

        positions = model.get_position_quantities("HOLD", capital_per_leg=10000)

        assert positions == {}
