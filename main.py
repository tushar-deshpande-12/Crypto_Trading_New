"""
Crypto AI Predictor - Main Entry Point
Unified application for cryptocurrency tracking, data fetching, and AI prediction
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import AppConfig

# Configure logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'crypto_ai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Launch the Crypto AI Predictor application (PyQt6)"""
    logger.info("=" * 70)
    logger.info(f"Starting {AppConfig.APP_NAME} v{AppConfig.VERSION}")
    logger.info("=" * 70)

    try:
        # Import PyQt6 application
        from PyQt6.QtWidgets import QApplication
        from src.gui_pyqt.app import CryptoAIPredictorApp

        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName(AppConfig.APP_NAME)
        app.setApplicationVersion(AppConfig.VERSION)

        # Create main window
        window = CryptoAIPredictorApp()
        window.show()

        logger.info("Application initialized successfully")

        # Start event loop
        sys.exit(app.exec())

    except ImportError as e:
        logger.error(f"PyQt6 not available: {e}")
        logger.error("Please install PyQt6: pip install PyQt6")
        print("\nError: PyQt6 is required to run this application.")
        print("Please install it with: pip install PyQt6")
        input("\nPress Enter to exit...")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    print("=" * 60)
    print(f"  {AppConfig.APP_NAME} v{AppConfig.VERSION}")
    print("  Launching unified application...")
    print("=" * 60)
    print()
    main()
