"""
Data Fetching Application Launcher
Standalone launcher for the cryptocurrency data pipeline GUI
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.gui.data_fetch_window import DataFetchWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fetch.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """
    Launch the data fetching application
    """
    logger.info("=" * 60)
    logger.info("Starting Crypto Data Pipeline")
    logger.info("=" * 60)

    try:
        app = DataFetchWindow()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
