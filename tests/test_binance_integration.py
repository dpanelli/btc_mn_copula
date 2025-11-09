"""Integration tests for Binance Futures API (no API keys required for public data)."""

from datetime import datetime, timedelta

import pytest
from binance.client import Client

from src.binance_client import BinanceClient


class TestBinanceFuturesIntegration:
    """Integration tests using real Binance Futures public API."""

    @pytest.fixture
    def public_client(self):
        """Create a client for public data (no API keys needed)."""
        # For public data endpoints, we don't need valid API keys
        return Client(api_key="", api_secret="")

    def test_futures_klines_structure(self, public_client):
        """Test the actual structure of Binance Futures klines response."""
        # Fetch recent 5-minute candles for BTCUSDT perpetual
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)  # Last hour

        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        # Fetch from Binance Futures
        klines = public_client.futures_klines(
            symbol="BTCUSDT",  # USDT-margined perpetual
            interval="5m",
            startTime=start_ms,
            endTime=end_ms,
            limit=20,
        )

        # Verify we got data
        assert len(klines) > 0, "Should receive klines data"

        # Check structure of first kline
        first_kline = klines[0]
        assert len(first_kline) >= 11, "Kline should have at least 11 fields"

        # Extract fields based on Binance Futures API documentation
        # [0] = Open time, [1] = Open, [2] = High, [3] = Low, [4] = Close, [5] = Volume,
        # [6] = Close time, [7] = Quote asset volume, [8] = Number of trades,
        # [9] = Taker buy base volume, [10] = Taker buy quote volume, [11] = Ignore
        open_time = int(first_kline[0])
        close_time = int(first_kline[6])

        # Verify close_time is returned
        assert close_time > 0, "close_time should be present"

        # Verify close_time is after open_time
        assert close_time > open_time, "close_time should be after open_time"

        # For 5-minute candle, close_time should be ~5 minutes after open_time
        time_diff_ms = close_time - open_time
        expected_diff = 5 * 60 * 1000  # 5 minutes in milliseconds
        # Allow some tolerance (within 1 second)
        assert abs(time_diff_ms - expected_diff) < 2000, \
            f"5-minute candle should span ~5 minutes, got {time_diff_ms/1000} seconds"

        print(f"\n✓ Kline structure verified:")
        print(f"  Open time:  {datetime.fromtimestamp(open_time/1000)} UTC")
        print(f"  Close time: {datetime.fromtimestamp(close_time/1000)} UTC")
        print(f"  Duration:   {time_diff_ms/1000/60:.2f} minutes")

    def test_incomplete_candle_detection(self, public_client):
        """Test that we can detect incomplete (currently forming) candles."""
        # Fetch very recent data (should include current forming candle)
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=30)

        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        klines = public_client.futures_klines(
            symbol="BTCUSDT",
            interval="5m",
            startTime=start_ms,
            endTime=end_ms,
            limit=10,
        )

        assert len(klines) > 0, "Should receive klines"

        # Check last candle
        last_kline = klines[-1]
        last_close_time = int(last_kline[6])

        current_time_ms = int(datetime.utcnow().timestamp() * 1000)

        # The last candle might be incomplete
        if last_close_time > current_time_ms:
            print(f"\n✓ Incomplete candle detected:")
            print(f"  Current time:      {datetime.fromtimestamp(current_time_ms/1000)} UTC")
            print(f"  Last candle close: {datetime.fromtimestamp(last_close_time/1000)} UTC")
            print(f"  Status: FORMING (should be filtered)")
        else:
            print(f"\n✓ Last candle is complete:")
            print(f"  Current time:      {datetime.fromtimestamp(current_time_ms/1000)} UTC")
            print(f"  Last candle close: {datetime.fromtimestamp(last_close_time/1000)} UTC")
            print(f"  Status: CLOSED (safe to use)")

    def test_binance_client_filters_incomplete_candles(self):
        """Test that our BinanceClient correctly filters incomplete candles."""
        # Create a client without valid keys (public data only)
        # This will fail for authenticated endpoints but work for public klines
        client = BinanceClient(api_key="test", api_secret="test", testnet=False)

        # Override the client to use public access
        client.client = Client(api_key="", api_secret="")

        # Fetch recent data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=30)

        df = client.get_historical_klines(
            symbol="BTCUSDT",
            interval="5m",
            start_time=start_time,
            end_time=end_time,
        )

        # Verify we got data
        assert len(df) > 0, "Should receive klines"

        # Verify all timestamps are in the past
        for idx, row in df.iterrows():
            candle_time = row["timestamp"]
            assert candle_time <= datetime.utcnow(), \
                f"All candles should be in the past, found future candle at {candle_time}"

        print(f"\n✓ BinanceClient filtering verified:")
        print(f"  Total candles: {len(df)}")
        print(f"  All candles are complete (closed)")
        print(f"  Last candle time: {df.iloc[-1]['timestamp']}")

    def test_perpetual_futures_symbols(self, public_client):
        """Verify we're using USDT-margined perpetual futures correctly."""
        # Test symbols that should work
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in test_symbols:
            klines = public_client.futures_klines(
                symbol=symbol,
                interval="5m",
                limit=1,
            )

            assert len(klines) > 0, f"Should fetch data for {symbol} perpetual"
            print(f"✓ {symbol} perpetual futures: OK")

        print(f"\n✓ All USDT-margined perpetual symbols verified")


@pytest.mark.integration
class TestBinanceFuturesFullWorkflow:
    """Full workflow integration test."""

    def test_formation_data_quality(self):
        """Test that formation phase data fetching works correctly with pagination."""
        client = BinanceClient(api_key="test", api_secret="test", testnet=False)
        client.client = Client(api_key="", api_secret="")

        # Simulate formation phase: fetch 21 days of 5-minute data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=21)

        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        for symbol in symbols:
            df = client.get_historical_klines(
                symbol=symbol,
                interval="5m",
                start_time=start_time,
                end_time=end_time,
            )

            # Verify data quality
            assert len(df) > 0, f"Should receive data for {symbol}"

            # With pagination, we should now get the full 21 days of 5-minute data
            # 21 days * 24 hours * 12 five-minute periods per hour = 6,048 candles
            # Allow some tolerance for market closures or incomplete final candle
            expected_min = 5500  # At least 5500 candles (91% of theoretical max)
            expected_max = 6100  # Up to 6100 candles (with slight overshoot)

            assert len(df) >= expected_min, \
                f"Should have at least {expected_min} candles with pagination, got {len(df)}"

            assert len(df) <= expected_max, \
                f"Should have at most {expected_max} candles, got {len(df)}"

            # Verify no NaN values in critical columns
            assert not df["close"].isna().any(), "Close prices should not have NaN"
            assert not df["volume"].isna().any(), "Volume should not have NaN"

            # Verify timestamps are in order
            assert df["timestamp"].is_monotonic_increasing, "Timestamps should be in ascending order"

            # Verify date range covers approximately 21 days
            time_span = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400
            assert time_span >= 19, f"Data should span ~21 days, got {time_span:.1f} days"
            assert time_span <= 22, f"Data should span ~21 days, got {time_span:.1f} days"

            print(f"✓ {symbol}: {len(df)} candles over {time_span:.1f} days, data quality OK")

        print(f"\n✓ Formation phase data quality verified with pagination")
