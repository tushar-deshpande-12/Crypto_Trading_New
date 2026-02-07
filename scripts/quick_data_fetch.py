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
    print("QUICK MULTI-TIMEFRAME DATA FETCH FOR TRAINING")
    print("=" * 80)
    print()
    print("Downloads 50,000 candles per timeframe for multiple timeframes")
    print()

    # Default symbols
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    candles = 50000
    timeframes = list(AppConfig.MULTI_TIMEFRAMES)

    print(f"Symbols:    {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Candles:    {candles:,} per timeframe")
    print(f"Total:      {candles * len(timeframes):,} candles per symbol")
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

    # Fetch data for each symbol across all timeframes
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Fetching {symbol} across {len(timeframes)} timeframes...")
        print("-" * 80)

        for tf_idx, tf in enumerate(timeframes):
            print(f"\n  [{tf_idx+1}/{len(timeframes)}] {symbol} @ {tf}...")

            # Switch fetcher interval
            data_manager.fetcher.interval = tf
            data_manager.interval = tf

            def progress(current, total, message):
                pct = (current / total * 100) if total > 0 else 0
                print(f"    Progress: {current:,}/{total:,} ({pct:.1f}%) - {message}")

            try:
                dataset_path = data_manager.fetch_and_save(
                    symbol=symbol,
                    max_candles=candles,
                    progress_callback=progress,
                    metadata={'interval': tf, 'timeframe': tf}
                )

                if dataset_path:
                    print(f"  [OK] {tf} saved to: {dataset_path}")
                else:
                    print(f"  [X] Failed to fetch {symbol} @ {tf}")

            except Exception as e:
                print(f"  [X] Error fetching {symbol} @ {tf}: {e}")
                import traceback
                traceback.print_exc()

    data_manager.close()

    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME DATA FETCH COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Run the main application: run.bat")
    print("2. Go to AI Training tab")
    print("3. Start training")
    print()


if __name__ == "__main__":
    main()
