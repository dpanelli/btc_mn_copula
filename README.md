# BTC Market-Neutral Copula Trading Bot

A sophisticated algorithmic trading bot implementing copula-based pairs trading for cryptocurrency futures markets. The strategy is based on the academic paper "Copula-Based Trading of Cointegrated Cryptocurrency Pairs" by Masood Tadi & Jiri Witzany (2023).

## Overview

This bot implements a **market-neutral** trading strategy that:
- Uses Bitcoin (BTC) as a reference asset to hedge altcoin exposure
- Identifies cointegrated spread pairs using statistical tests
- Models dependence structure using Gaussian copulas
- Generates trading signals based on conditional probabilities
- Trades on Binance Futures with configurable leverage

### Key Features

- **Automated Formation Phase**: Weekly selection of top cointegrated pairs
- **Real-time Trading**: 5-minute interval signal generation and execution
- **Risk Management**: Position sizing, leverage control, and stop-loss
- **State Persistence**: Saves copula parameters and trading state
- **Comprehensive Logging**: Detailed logs for monitoring and debugging

## Strategy Details

### Formation Phase (Weekly)

1. Fetch 21 days of historical 5-minute OHLCV data
2. Calculate spreads for all altcoin pairs: `S_i(t) = BTC(t) - beta_i * ALT_i(t)`
3. Test cointegration using both Engle-Granger (ADF) and Kapetanios-Shin-Snell (KSS) tests
4. Rank cointegrated pairs by Kendall's Tau correlation
5. Select top pair and fit Gaussian copula

### Trading Phase (Every 5 Minutes)

1. Fetch current prices for BTC and selected altcoins
2. Calculate conditional probabilities: `h_1|2` and `h_2|1`
3. Generate signals based on thresholds:
   - **LONG S1, SHORT S2**: when `h_1|2 < alpha1` AND `h_2|1 > 1-alpha1`
   - **SHORT S1, LONG S2**: when `h_1|2 > 1-alpha1` AND `h_2|1 < alpha1`
   - **CLOSE**: when both probabilities near 0.5 (within alpha2)
4. Execute market orders on Binance Futures

### Expected Performance

Based on the original paper (2-year backtest):
- **Annualized Return**: 37-76%
- **Sharpe Ratio**: 0.97-3.77
- **Win Rate**: 55-67%
- **Max Drawdown**: ~25-35%

**WARNING**: Past performance does not guarantee future results. Always test thoroughly before live trading.

## Project Structure

```
btc_mn_copulas/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging setup
│   ├── binance_client.py      # Binance API wrapper
│   ├── copula_model.py        # Copula calculations and signals
│   ├── formation.py           # Weekly formation phase
│   ├── trading.py             # Trading execution
│   └── state_manager.py       # State persistence
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_copula_model.py
│   └── test_state_manager.py
├── logs/                      # Log files (auto-generated)
├── state/                     # State persistence (auto-generated)
├── main.py                    # Main orchestrator
├── pyproject.toml             # Dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python >= 3.12
- Binance Futures account (or Testnet account)
- API keys with Futures trading permissions

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd btc_mn_copulas
```

2. **Install dependencies**:
```bash
# Using uv (recommended)
# 1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
#    curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies
uv sync
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

4. **Get Binance API keys**:
   - **Testnet** (recommended for testing): https://testnet.binancefuture.com/
   - **Live**: https://www.binance.com/en/my/settings/api-management

## Configuration

Edit `.env` file with your settings:

```bash
# API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
USE_TESTNET=true  # Set to false for live trading

# Trading Parameters
CAPITAL_PER_LEG=20000        # USDT per leg
MAX_LEVERAGE=3               # 1-3x recommended
ENTRY_THRESHOLD=0.10         # alpha1 (0.10 = 10th percentile)
EXIT_THRESHOLD=0.10          # alpha2

# Altcoin Universe
ALTCOINS=ETHUSDT,BNBUSDT,ADAUSDT,XRPUSDT,SOLUSDT,AVAXUSDT

# Formation & Trading Schedule
FORMATION_DAYS=21            # Days of data for formation
TRADING_INTERVAL_MINUTES=5   # Trading frequency
```

## Usage

### Running the Bot

**Testnet (recommended first)**:
```bash
# Ensure USE_TESTNET=true in .env
python main.py
```

**Live Trading**:
```bash
# Set USE_TESTNET=false in .env
python main.py
```

### Backtesting

You can run backtests without any API keys! The backtester automatically downloads high-quality **5-minute** data from Binance Vision (public S3 bucket).

```bash
# Basic run (defaults to Q1 2025 if not specified, but arguments are required)
python backtest.py --start 2025-01 --end 2025-03

# Custom date range and capital
python backtest.py --start 2024-01 --end 2024-06 --initial-capital 50000

# Run in parallel mode (faster for long periods)
python backtest.py --start 2024-01 --end 2024-12 --parallel
```

- **No API Keys Needed**: Uses public data.
- **Data Source**: Binance Vision (official historical data).
- **Interval**: 5-minute candles.
- **Output**: Generates a tearsheet report in `backtest_results/tearsheet/report.html`.

The bot will:
1. Initialize all components
2. Load existing formation state (if available)
3. Schedule weekly formation phase (Mondays 00:00 UTC)
4. Schedule 5-minute trading cycles
5. Run continuously until stopped (Ctrl+C)

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_copula_model.py -v
```

### Manual Formation Phase

To manually trigger a formation phase without waiting for the schedule:

```python
from src.config import get_config
from src.binance_client import BinanceClient
from src.formation import FormationManager

config = get_config()
client = BinanceClient(config.binance.api_key, config.binance.api_secret, testnet=True)
formation = FormationManager(client, config.trading.altcoins, config.trading.formation_days)

# Run formation
spread_pair = formation.run_formation()
print(f"Selected pair: {spread_pair.alt1} - {spread_pair.alt2}")
print(f"Parameters: beta1={spread_pair.beta1:.6f}, beta2={spread_pair.beta2:.6f}, rho={spread_pair.rho:.4f}")
```

## Monitoring

### Logs

Logs are written to `logs/trading.log` with rotation:
```bash
# View live logs
tail -f logs/trading.log

# Search for signals
grep "SIGNAL" logs/trading.log

# Check errors
grep "ERROR" logs/trading.log
```

### Trade Log

All trades are logged to `state/trade_log.jsonl`:
```bash
# View recent trades
tail -n 20 state/trade_log.jsonl | jq .

# Count successful trades
grep '"status":"success"' state/trade_log.jsonl | wc -l
```

### State Inspection

Check current state:
```python
from src.state_manager import StateManager

manager = StateManager()
summary = manager.get_state_summary()
print(summary)
```

## Risk Management

**IMPORTANT WARNINGS**:

1. **Start with Testnet**: Always test thoroughly on Binance Futures Testnet before live trading
2. **Capital at Risk**: Only trade with capital you can afford to lose
3. **Leverage Risk**: Higher leverage amplifies both gains and losses
4. **Market Conditions**: Strategy performance varies with market conditions
5. **Fees**: Account for trading fees (0.04% taker, 0.02% maker on Binance)
6. **API Security**: Never share API keys; use IP whitelisting
7. **Monitoring**: Always monitor the bot; don't run unattended initially

### Recommended Risk Settings

- Start with **low capital** (e.g., $1,000 per leg)
- Use **low leverage** (1-2x)
- Monitor **daily drawdown** and set stop-loss if needed
- Run **paper trading** for at least 2 weeks first

## Troubleshooting

### Common Issues

**1. API Key Error**:
```
ValueError: BINANCE_API_KEY and BINANCE_API_SECRET must be set
```
**Solution**: Create `.env` file from `.env.example` and add your API keys

**2. No Cointegrated Pairs**:
```
Formation phase failed - no suitable pairs found!
```
**Solution**: Market conditions may not have cointegrated pairs. Wait for next formation cycle or adjust altcoin universe

**3. Insufficient Balance**:
```
Error placing market order: Insufficient balance
```
**Solution**: Reduce `CAPITAL_PER_LEG` or deposit more USDT to Futures wallet

**4. Position Already Open**:
**Solution**: Bot tracks position state in `state/state.json`. Clear state if needed: `rm state/state.json`

## Development

### Running in Development Mode

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Format code (optional)
black src/ tests/
```

### Adding New Copula Families

The current implementation uses Gaussian copula. To add Student-t, Clayton, or other copulas:

1. Implement copula CDF and conditional CDF in `src/copula_model.py`
2. Add parameter estimation method
3. Update `FormationManager` to test multiple copulas and select by AIC
4. Add tests in `tests/test_copula_model.py`

## References

- **Original Paper**: Tadi, M., & Witzany, J. (2023). "Copula-Based Trading of Cointegrated Cryptocurrency Pairs"
- **Binance Futures API**: https://binance-docs.github.io/apidocs/futures/en/
- **Copula Theory**: Nelsen, R. B. (2006). "An Introduction to Copulas"
- **Cointegration**: Engle, R. F., & Granger, C. W. (1987). "Co-integration and error correction"


