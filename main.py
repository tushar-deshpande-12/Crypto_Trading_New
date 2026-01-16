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
    """Launch the Crypto AI Predictor application"""
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
        # Fall back to Tkinter if PyQt6 is not installed
        logger.warning(f"PyQt6 not available ({e}), falling back to Tkinter GUI")
        try:
            import tkinter as tk
            from src.gui.app import CryptoAIPredictorApp as TkinterApp

            root = tk.Tk()
            app = TkinterApp(root)
            logger.info("Tkinter application initialized")
            app.run()
            app.cleanup()

        except Exception as tk_error:
            logger.error(f"Tkinter GUI also failed: {tk_error}", exc_info=True)
            raise

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
