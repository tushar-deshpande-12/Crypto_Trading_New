"""
Test script to verify all imports work correctly
Run this to test the application without launching the GUI
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports"""

    print("Testing imports...")
    print("-" * 60)

    try:
        # Core
        print("[OK] Importing core.config...", end=" ")
        from src.core.config import AppConfig
        print(f"v{AppConfig.VERSION}")

        # API
        print("[OK] Importing api.binance_client...", end=" ")
        from src.api.binance_client import BinanceAPIClient
        print("OK")

        # Data
        print("[OK] Importing data.fetcher...", end=" ")
        from src.data.fetcher import CryptoDataFetcher
        print("OK")

        print("[OK] Importing data.storage...", end=" ")
        from src.data.storage import DataStorage
        print("OK")

        print("[OK] Importing data.manager...", end=" ")
        from src.data.manager import DataManager
        print("OK")

        print("[OK] Importing data.config...", end=" ")
        from src.data.config import DataPipelineConfig, ConfigPresets
        print("OK")

        # GUI
        print("[OK] Importing gui.styles...", end=" ")
        from src.gui.styles import setup_styles, get_color
        print("OK")

        print("[OK] Importing gui.components...", end=" ")
        from src.gui.components import SymbolTable, ChartPanel, DataPanel
        print("OK")

        print("[OK] Importing gui.app...", end=" ")
        from src.gui.app import CryptoAIPredictorApp
        print("OK")

        # Utils
        print("[OK] Importing utils.formatters...", end=" ")
        from src.utils.formatters import format_price, format_volume
        print("OK")

        print("-" * 60)
        print("SUCCESS: All imports successful!")
        print()
        print(f"Application: {AppConfig.APP_NAME}")
        print(f"Version: {AppConfig.VERSION}")
        print(f"Dataset Directory: {AppConfig.DATASET_DIR}")
        print(f"Default Interval: {AppConfig.DEFAULT_INTERVAL}")
        print(f"Window Size: {AppConfig.WINDOW_WIDTH}x{AppConfig.WINDOW_HEIGHT}")
        print()
        print("Ready to launch! Run: python app.py")

        return True

    except ImportError as e:
        print(f"\n[ERROR] Import Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_manager():
    """Test data manager initialization"""
    print("\nTesting DataManager initialization...")
    print("-" * 60)

    try:
        from src.data import DataManager

        manager = DataManager(storage_dir="dataset", interval="1h")
        print("[OK] DataManager initialized")

        stats = manager.get_storage_statistics()
        print(f"[OK] Storage stats: {stats['total_datasets']} datasets, {stats['total_size_mb']} MB")

        manager.close()
        print("[OK] DataManager closed")

        print("SUCCESS: DataManager test passed!")
        return True

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_client():
    """Test API client initialization"""
    print("\nTesting BinanceAPIClient initialization...")
    print("-" * 60)

    try:
        from src.api.binance_client import BinanceAPIClient

        client = BinanceAPIClient()
        print("[OK] BinanceAPIClient initialized")

        # Don't actually call API in test (requires internet)
        print("[OK] API client ready (skipping actual API calls)")

        client.close()
        print("[OK] API client closed")

        print("SUCCESS: API client test passed!")
        return True

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  Crypto AI Predictor - Import Test Suite")
    print("=" * 60)
    print()

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test data manager
    results.append(("DataManager", test_data_manager()))

    # Test API client
    results.append(("API Client", test_api_client()))

    # Summary
    print()
    print("=" * 60)
    print("  Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name:20} [{status}]")

    print("=" * 60)

    if all(result[1] for result in results):
        print("\nSUCCESS: All tests passed! Application is ready to run.")
        print("\nTo launch the application:")
        print("  Windows: run.bat")
        print("  Python:  python app.py")
    else:
        print("\nWARNING: Some tests failed. Please check the errors above.")
        sys.exit(1)
