"""
Quick script to fetch sufficient data for training
Run this before training if you need more data
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.manager import DataManager
from src.core.config import AppConfig

def main():
    print("=" * 80)
    print("QUICK DATA FETCH FOR TRAINING")
    print("=" * 80)
    print()
    print("This will download enough data for volatility-enhanced training")
    print("Recommended: 5,000-50,000 candles per symbol")
    print()

    # Default symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    candles = 5000  # Minimum for volatility features

    print(f"Symbols: {', '.join(symbols)}")
    print(f"Candles per symbol: {candles:,}")
    print()

    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Initialize data manager
    data_manager = DataManager(
        storage_dir=AppConfig.DATASET_DIR,
        interval="1h"
    )

    # Fetch data for each symbol
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Fetching {symbol}...")
        print("-" * 80)

        def progress(current, total, message):
            pct = (current / total * 100) if total > 0 else 0
            print(f"  Progress: {current:,}/{total:,} ({pct:.1f}%) - {message}")

        try:
            dataset_path = data_manager.fetch_and_save(
                symbol=symbol,
                max_candles=candles,
                progress_callback=progress
            )

            if dataset_path:
                print(f"[OK] Saved to: {dataset_path}")
            else:
                print(f"[X] Failed to fetch {symbol}")

        except Exception as e:
            print(f"[X] Error fetching {symbol}: {e}")
            import traceback
            traceback.print_exc()

    data_manager.close()

    print("\n" + "=" * 80)
    print("DATA FETCH COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Run the main application: run.bat")
    print("2. Go to AI Training tab")
    print("3. Start training")
    print()


if __name__ == "__main__":
    main()
