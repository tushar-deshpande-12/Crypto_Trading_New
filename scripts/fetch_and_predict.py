"""
Fetch latest market data and generate 10-hour predictions
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.fetcher import CryptoDataFetcher
from src.data.storage import DataStorage
from src.ml.inference.predictor import CryptoPredictor


def main():
    print("=" * 80)
    print("CRYPTOCURRENCY PREDICTION SYSTEM")
    print("Fetching latest data and generating 10-hour predictions")
    print("=" * 80)
    print()

    # Configuration
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    model_path = 'models/checkpoints/best_model.ckpt'
    scaler_path = 'models/scalers.pkl'
    dataset_dir = 'dataset'

    # Step 1: Fetch latest market data
    print("\n[STEP 1/3] FETCHING LATEST MARKET DATA")
    print("-" * 80)

    fetcher = CryptoDataFetcher(interval="1h")
    storage = DataStorage(base_dir=dataset_dir)

    for symbol in symbols:
        try:
            print(f"\n[{symbol}] Fetching latest 200 hours of data...")

            # Fetch latest 200 hours (provides enough context)
            data = fetcher.fetch_latest_ohlcv(symbol, hours_back=200)

            if data:
                print(f"[{symbol}] Fetched {len(data)} candles")
                print(f"[{symbol}] Latest timestamp: {data[-1]['datetime']}")

                # Save to dataset
                saved_path = storage.save_dataset(
                    symbol=symbol,
                    data=data,
                    metadata={
                        'fetch_type': 'latest_update',
                        'candles_fetched': len(data),
                        'fetch_timestamp': datetime.now().isoformat()
                    }
                )
                print(f"[{symbol}] Saved to: {saved_path}")
            else:
                print(f"[{symbol}] X Failed to fetch data")

        except Exception as e:
            print(f"[{symbol}] X Error: {e}")
            import traceback
            traceback.print_exc()

    fetcher.close()

    # Step 2: Load model and generate predictions
    print("\n\n[STEP 2/3] LOADING MODEL AND GENERATING PREDICTIONS")
    print("-" * 80)

    try:
        # Initialize predictor
        predictor = CryptoPredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            dataset_dir=dataset_dir,
            verbose=True
        )

        # Generate predictions for all symbols
        print("\n\n[STEP 3/3] GENERATING 10-HOUR PREDICTIONS")
        print("-" * 80)

        all_predictions = {}

        for symbol in symbols:
            try:
                print(f"\n{'=' * 80}")
                print(f"PREDICTING: {symbol}")
                print('=' * 80)

                predictions = predictor.predict(
                    symbol=symbol,
                    n_hours=10,
                    return_confidence_intervals=True
                )

                all_predictions[symbol] = predictions

                # Display results
                print(f"\n[{symbol}] PREDICTION RESULTS")
                print("-" * 80)

                current_time = datetime.now()

                for i, (timestamp, median, lower_95, upper_95) in enumerate(zip(
                    predictions['timestamps'],
                    predictions['median'],
                    predictions['lower_95'],
                    predictions['upper_95']
                ), 1):
                    print(f"Hour {i:2d} | {timestamp.strftime('%Y-%m-%d %H:%M')} | "
                          f"Median: ${median:,.2f} | "
                          f"95% CI: [${lower_95:,.2f} - ${upper_95:,.2f}]")

                # Export to CSV
                output_file = f"prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                predictor.export_prediction(predictions, output_file)
                print(f"\n[{symbol}] SUCCESS - Predictions exported to: {output_file}")

            except Exception as e:
                print(f"\n[{symbol}] ERROR - Prediction failed: {e}")
                import traceback
                traceback.print_exc()

        # Summary
        print("\n\n" + "=" * 80)
        print("PREDICTION SUMMARY")
        print("=" * 80)

        for symbol, predictions in all_predictions.items():
            if predictions:
                median = predictions['median']
                print(f"\n{symbol}:")
                print(f"  Current to +10h price range: ${median.min():,.2f} - ${median.max():,.2f}")
                print(f"  Expected change: ${median[-1] - median[0]:+,.2f} "
                      f"({((median[-1] / median[0]) - 1) * 100:+.2f}%)")

        print("\n" + "=" * 80)
        print("SUCCESS - PREDICTION COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\nX FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    print("\n\nPress Enter to exit...")
    input()
    sys.exit(exit_code)
