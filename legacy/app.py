"""
Crypto AI Predictor - Main Launcher
Unified application for cryptocurrency tracking, data fetching, and prediction
"""

import sys
import tkinter as tk
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.app import CryptoAIPredictorApp
from src.core.config import AppConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crypto_ai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Launch the Crypto AI Predictor application"""
    logger.info("=" * 70)
    logger.info(f"Starting {AppConfig.APP_NAME} v{AppConfig.VERSION}")
    logger.info("=" * 70)

    try:
        # Create root window
        root = tk.Tk()

        # Create and run application
        app = CryptoAIPredictorApp(root)
        logger.info("Application initialized successfully")

        # Start main loop
        app.run()

        # Cleanup
        app.cleanup()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
