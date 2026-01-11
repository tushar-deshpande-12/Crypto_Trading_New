"""
Example: How to use the Data Pipeline programmatically
Demonstrates various use cases for the cryptocurrency data pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data import DataManager, ConfigPresets


def example_basic_fetch():
    """Example 1: Basic data fetching"""
    print("=" * 60)
    print("Example 1: Fetch Historical Data for Bitcoin")
    print("=" * 60)

    # Initialize data manager
    manager = DataManager(storage_dir="dataset", interval="1h")

    # Fetch 10,000 candles (~13.7 months of 1h data)
    print("Fetching BTC data...")
    dataset_path = manager.fetch_and_save(
        symbol="BTC",
        max_candles=10000
    )

    if dataset_path:
        print(f"\n✓ Success! Data saved to: {dataset_path}")
    else:
        print("\n✗ Failed to fetch data")

    manager.close()


def example_load_data():
    """Example 2: Load existing dataset"""
    print("\n" + "=" * 60)
    print("Example 2: Load Existing Dataset")
    print("=" * 60)

    manager = DataManager(storage_dir="dataset", interval="1h")

    # Load latest BTC data
    data = manager.load_latest("BTCUSDT")

    if data:
        print(f"\nLoaded {len(data)} candles")
        print("\nFirst 5 candles:")
        for candle in data[:5]:
            print(f"  {candle['datetime']}: "
                  f"O={candle['open']:.2f}, "
                  f"H={candle['high']:.2f}, "
                  f"L={candle['low']:.2f}, "
                  f"C={candle['close']:.2f}, "
                  f"V={candle['volume']:.2f}")
    else:
        print("\nNo data found. Run example_basic_fetch() first.")

    manager.close()


def example_check_freshness():
    """Example 3: Check data freshness"""
    print("\n" + "=" * 60)
    print("Example 3: Check Data Freshness")
    print("=" * 60)

    manager = DataManager(storage_dir="dataset", interval="1h")

    # Check if BTC data needs updating (24h threshold)
    freshness = manager.check_data_freshness("BTCUSDT", time_threshold_hours=24)

    print(f"\nSymbol: {freshness['symbol']}")
    print(f"Has Data: {freshness['has_data']}")

    if freshness['has_data']:
        print(f"Last Fetch: {freshness.get('fetch_timestamp', 'Unknown')}")
        print(f"Candles: {freshness.get('candle_count', 0):,}")
        print(f"Should Fetch New: {freshness['should_fetch_new']}")

        if freshness['should_fetch_new']:
            print("\n→ Data is old, fetching new data...")
            manager.fetch_and_save("BTC", max_candles=10000)
        else:
            print("\n→ Data is fresh, using existing dataset")
    else:
        print("\nNo existing data. Fetching new...")
        manager.fetch_and_save("BTC", max_candles=10000)

    manager.close()


def example_multiple_symbols():
    """Example 4: Fetch multiple cryptocurrencies"""
    print("\n" + "=" * 60)
    print("Example 4: Fetch Multiple Symbols")
    print("=" * 60)

    manager = DataManager(storage_dir="dataset", interval="1h")

    symbols = ["BTC", "ETH", "BNB", "SOL"]
    print(f"\nFetching data for: {', '.join(symbols)}")

    def progress_callback(symbol, current, total, message):
        print(f"  [{current}/{total}] {message}")

    results = manager.fetch_and_save_multiple(
        symbols=symbols,
        max_candles=5000,
        progress_callback=progress_callback
    )

    print("\nResults:")
    for symbol, path in results.items():
        if path:
            print(f"  ✓ {symbol}: {path.name}")
        else:
            print(f"  ✗ {symbol}: Failed")

    manager.close()


def example_list_datasets():
    """Example 5: List all datasets"""
    print("\n" + "=" * 60)
    print("Example 5: List All Datasets")
    print("=" * 60)

    manager = DataManager(storage_dir="dataset", interval="1h")

    datasets = manager.list_all_datasets()

    if datasets:
        print(f"\nFound {len(datasets)} datasets:\n")
        for ds in datasets:
            symbol = ds.get('symbol', 'Unknown')
            candles = ds.get('candle_count', 0)
            timestamp = ds.get('fetch_timestamp', 'Unknown')
            print(f"  {symbol:12} | {candles:6,} candles | {timestamp}")
    else:
        print("\nNo datasets found")

    # Show storage statistics
    stats = manager.get_storage_statistics()
    print(f"\nStorage Statistics:")
    print(f"  Total Datasets: {stats['total_datasets']}")
    print(f"  Total Symbols: {stats['total_symbols']}")
    print(f"  Total Size: {stats['total_size_mb']:.2f} MB")

    manager.close()


def example_cleanup():
    """Example 6: Cleanup old datasets"""
    print("\n" + "=" * 60)
    print("Example 6: Cleanup Old Datasets")
    print("=" * 60)

    manager = DataManager(storage_dir="dataset", interval="1h")

    # Keep only 3 most recent datasets for BTC
    print("\nCleaning up old BTCUSDT datasets (keeping 3 most recent)...")
    deleted = manager.cleanup_old_datasets("BTCUSDT", keep_latest=3)

    print(f"Deleted {deleted} old datasets")

    manager.close()


def example_custom_config():
    """Example 7: Using custom configuration"""
    print("\n" + "=" * 60)
    print("Example 7: Custom Configuration")
    print("=" * 60)

    # Use ML training preset
    config = ConfigPresets.ml_training()
    print(f"\nUsing ML Training Config:")
    print(f"  Max Candles: {config.default_max_candles:,}")
    print(f"  CSV Format: {config.save_csv_format}")
    print(f"  JSON Format: {config.save_json_format}")

    manager = DataManager(
        storage_dir="ml_dataset",
        interval="1h"
    )

    print("\nThis configuration is optimized for ML training")
    print("(CSV format, 20,000 candles, duplicate removal)")

    manager.close()


def example_convert_to_pandas():
    """Example 8: Convert to pandas DataFrame"""
    print("\n" + "=" * 60)
    print("Example 8: Convert to Pandas DataFrame")
    print("=" * 60)

    try:
        import pandas as pd
    except ImportError:
        print("\nPandas not installed. Install with: pip install pandas")
        return

    manager = DataManager(storage_dir="dataset", interval="1h")

    data = manager.load_latest("BTCUSDT")

    if data:
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        print(f"\nDataFrame Info:")
        print(df.info())

        print(f"\nFirst 5 rows:")
        print(df[['open', 'high', 'low', 'close', 'volume']].head())

        print(f"\nStatistics:")
        print(df[['open', 'high', 'low', 'close', 'volume']].describe())
    else:
        print("\nNo data found. Run example_basic_fetch() first.")

    manager.close()


def main():
    """Run all examples"""
    print("""
╔════════════════════════════════════════════════════════════╗
║     Cryptocurrency Data Pipeline Examples                  ║
║     Professional Data Fetching for AI Prediction           ║
╚════════════════════════════════════════════════════════════╝
    """)

    examples = [
        ("1", "Fetch Historical Data", example_basic_fetch),
        ("2", "Load Existing Dataset", example_load_data),
        ("3", "Check Data Freshness", example_check_freshness),
        ("4", "Fetch Multiple Symbols", example_multiple_symbols),
        ("5", "List All Datasets", example_list_datasets),
        ("6", "Cleanup Old Datasets", example_cleanup),
        ("7", "Custom Configuration", example_custom_config),
        ("8", "Convert to Pandas", example_convert_to_pandas),
    ]

    print("Available Examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print("  0. Run All Examples")

    choice = input("\nSelect example (0-8): ").strip()

    if choice == "0":
        for num, name, func in examples:
            func()
    elif choice in [num for num, _, _ in examples]:
        for num, name, func in examples:
            if choice == num:
                func()
                break
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
