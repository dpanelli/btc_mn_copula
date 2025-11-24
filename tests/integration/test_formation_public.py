import logging
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.binance_client import BinanceClient
from src.formation import FormationManager
from src.logger import setup_logger

def test_formation_public_data():
    """
    Test that formation phase runs correctly using public data (no API keys).
    This verifies that:
    1. BinanceClient works without keys for public endpoints
    2. FormationManager can fetch data and calculate statistics
    3. No attribute errors occur during the process
    """
    # Setup logging
    setup_logger("test_formation", "test_formation.log", "INFO")
    logger = logging.getLogger("test_formation")
    
    logger.info("Starting public data formation test...")
    
    # 1. Initialize BinanceClient without keys
    # This forces it to use public endpoints
    client = BinanceClient(api_key=None, api_secret=None, testnet=False)
    
    # 2. Initialize FormationManager with a small subset of coins for speed
    # Using liquid pairs that likely have history
    altcoins = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
    
    formation_manager = FormationManager(
        binance_client=client,
        altcoins=altcoins,
        formation_days=5,  # Short period for faster test
        interval="1h"      # 1h interval for faster data fetching
    )
    
    # 3. Run formation
    logger.info("Running formation...")
    try:
        spread_pair = formation_manager.run_formation()
        
        if spread_pair:
            logger.info(f"SUCCESS: Formation found pair: {spread_pair.alt1}-{spread_pair.alt2}")
            logger.info(f"Params: beta1={spread_pair.beta1:.4f}, beta2={spread_pair.beta2:.4f}, rho={spread_pair.rho:.4f}")
        else:
            logger.info("SUCCESS: Formation ran but found no suitable pairs (expected with small subset)")
            
    except Exception as e:
        logger.error(f"FAILURE: Formation raised exception: {e}")
        raise

if __name__ == "__main__":
    test_formation_public_data()
