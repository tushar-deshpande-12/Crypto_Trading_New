"""
QThread Workers for Async Operations

Provides thread workers for long-running operations like:
- Market data fetching
- Historical data downloading
- Model training
- Predictions
- Backtesting
"""

from PyQt6.QtCore import QThread, pyqtSignal
import traceback
import pandas as pd
from typing import Optional, Dict, List, Any


class MarketDataWorker(QThread):
    """
    Worker for fetching market data from Binance API.

    Emits:
        result: List of market data dicts
        error: Error message string
        finished: When work completes
    """
    result = pyqtSignal(list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client

    def run(self):
        try:
            data = self.api_client.get_usdt_pairs_detailed()
            if data:
                self.result.emit(data)
            else:
                self.error.emit("Failed to fetch market data")
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class StockMarketDataWorker(QThread):
    """
    Worker for fetching market data from NSE India API.

    Emits:
        result: List of stock market data dicts
        error: Error message string
        finished: When work completes
    """
    result = pyqtSignal(list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client

    def run(self):
        try:
            data = self.api_client.get_inr_pairs_detailed()
            if data:
                self.result.emit(data)
            else:
                self.error.emit("Failed to fetch NSE stock market data")
        except Exception as e:
            self.error.emit(f"NSE API Error: {str(e)}")
        finally:
            self.finished.emit()


class ChartDataWorker(QThread):
    """
    Worker for fetching chart/candlestick data.

    Fetches more data (2000 candles default) to support multi-timescale analysis.
    Will try to use locally stored data first if available, then fall back to API.

    Emits:
        result: List of chart data dicts
        error: Error message string
        finished: When work completes
    """
    result = pyqtSignal(list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    # Default limit increased for multi-timescale support
    # 2000 1h candles = ~83 days of data, good for daily/weekly views
    DEFAULT_LIMIT = 2000

    def __init__(self, api_client, symbol: str, interval: str = "1h", limit: int = None,
                 data_manager=None):
        super().__init__()
        self.api_client = api_client
        self.symbol = symbol
        self.interval = interval
        self.limit = limit if limit is not None else self.DEFAULT_LIMIT
        self.data_manager = data_manager

    def run(self):
        try:
            chart_data = None

            # Try to load from local storage first (faster and more data)
            if self.data_manager:
                try:
                    local_data = self._load_local_data()
                    if local_data and len(local_data) >= self.limit:
                        chart_data = local_data[-self.limit:]  # Use most recent data
                except Exception:
                    pass  # Fall through to API

            # Fall back to API if no local data
            if not chart_data:
                chart_data = self.api_client.get_klines_formatted(
                    self.symbol, self.interval, self.limit
                )

            if chart_data:
                self.result.emit(chart_data)
            else:
                self.error.emit(f"Failed to load chart for {self.symbol}")

        except Exception as e:
            self.error.emit(f"Chart error for {self.symbol}: {e}")
        finally:
            self.finished.emit()

    def _load_local_data(self) -> Optional[List[Dict]]:
        """Try to load chart data from local storage"""
        from pathlib import Path
        import json
        import pandas as pd

        # Try to find dataset directory for this symbol
        dataset_dir = Path("dataset") / self.symbol
        if not dataset_dir.exists():
            return None

        # Find the most recent dataset with most data
        best_dir = None
        best_count = 0

        for sub_dir in dataset_dir.iterdir():
            if sub_dir.is_dir():
                meta_path = sub_dir / "metadata.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        count = meta.get('candle_count', 0)
                        if count > best_count:
                            best_count = count
                            best_dir = sub_dir
                    except Exception:
                        continue

        if not best_dir:
            return None

        # Load data
        data_path = best_dir / "data.json"
        if data_path.exists():
            with open(data_path, 'r') as f:
                data = json.load(f)

            # Convert to chart format
            chart_data = []
            for row in data:
                chart_data.append({
                    'datetime': row.get('datetime', row.get('timestamp')),
                    'open': float(row.get('open', 0)),
                    'high': float(row.get('high', 0)),
                    'low': float(row.get('low', 0)),
                    'close': float(row.get('close', 0)),
                    'volume': float(row.get('volume', 0))
                })

            return chart_data

        return None


class CCXTChartWorker(QThread):
    """
    Worker for fetching chart data using CCXT.

    Supports all timeframes (1m to 1w) and maximum candle fetching.

    Emits:
        progress: (current, total, message) during fetch
        result: DataFrame with OHLCV data
        error: Error message string
        finished: When work completes
    """
    progress = pyqtSignal(int, int, str)
    result = pyqtSignal(object)  # pandas DataFrame
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, symbol: str, timeframe: str = '1h', limit: int = 500,
                 exchange_id: str = 'binance'):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.limit = limit
        self.exchange_id = exchange_id
        self._stopped = False

    def run(self):
        try:
            from src.api.ccxt_client import get_ccxt_client

            self.progress.emit(0, 100, f"Connecting to {self.exchange_id}...")

            # Get cached CCXT client (avoids reconnecting every time)
            client = get_ccxt_client(self.exchange_id)

            # Convert symbol format if needed (BTCUSDT -> BTC/USDT)
            ccxt_symbol = client.convert_symbol_format(self.symbol, to_ccxt=True)

            self.progress.emit(20, 100, f"Fetching {self.limit} candles for {ccxt_symbol}...")

            def progress_cb(current, total, message):
                if not self._stopped:
                    pct = 20 + int((current / max(total, 1)) * 70)
                    self.progress.emit(pct, 100, message)

            # Fetch data with pagination for large requests
            if self.limit > 1000:
                df = client.get_max_ohlcv(
                    ccxt_symbol,
                    self.timeframe,
                    max_candles=self.limit,
                    progress_callback=progress_cb
                )
            else:
                df = client.get_ohlcv(ccxt_symbol, self.timeframe, self.limit)

            if self._stopped:
                return

            if df is not None and len(df) > 0:
                self.progress.emit(100, 100, f"Loaded {len(df)} candles")
                self.result.emit(df)
            else:
                self.error.emit(f"No data returned for {ccxt_symbol}")

        except Exception as e:
            self.error.emit(f"CCXT Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.finished.emit()

    def stop(self):
        """Request stop."""
        self._stopped = True


class DataFetchWorker(QThread):
    """
    Worker for downloading historical data across multiple timeframes.

    Downloads 50K candles for each timeframe (5m, 15m, 30m, 1h, 4h) by default.

    Emits:
        progress: (current, total, message) during download
        result: Success/failure dict
        error: Error message string
        finished: When work completes
    """
    progress = pyqtSignal(int, int, str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, api_client, data_manager, symbol: str, max_candles: int):
        super().__init__()
        self.api_client = api_client
        self.data_manager = data_manager
        self.symbol = symbol
        self.max_candles = max_candles
        self._stopped = False

    def run(self):
        try:
            from src.core.config import AppConfig
            timeframes = list(AppConfig.MULTI_TIMEFRAMES)
            total_tf = len(timeframes)
            tf_results = {}

            self.progress.emit(0, 100, f"Fetching {self.symbol} across {total_tf} timeframes ({', '.join(timeframes)})...")

            for tf_idx, tf in enumerate(timeframes):
                if self._stopped:
                    break

                base_pct = int((tf_idx / total_tf) * 100)
                tf_pct_range = int(100 / total_tf)

                self.progress.emit(base_pct, 100, f"[{tf_idx+1}/{total_tf}] Fetching {self.symbol} @ {tf} ({self.max_candles:,} candles)...")

                # Switch fetcher to this timeframe
                self.data_manager.fetcher.interval = tf
                original_interval = self.data_manager.interval
                self.data_manager.interval = tf

                try:
                    def tf_progress(current, total, message):
                        if not self._stopped:
                            # Scale progress within this timeframe's slice
                            if total > 0:
                                sub_pct = int((current / total) * tf_pct_range)
                            else:
                                sub_pct = 0
                            self.progress.emit(base_pct + sub_pct, 100, f"[{tf}] {message}")

                    path = self.data_manager.fetch_and_save(
                        symbol=self.symbol,
                        max_candles=self.max_candles,
                        progress_callback=tf_progress,
                        metadata={'interval': tf, 'timeframe': tf}
                    )
                    tf_results[tf] = 'OK' if path else 'FAILED'
                except Exception as e:
                    tf_results[tf] = f'ERROR: {e}'
                finally:
                    self.data_manager.interval = original_interval
                    self.data_manager.fetcher.interval = original_interval

            if self._stopped:
                self.result.emit({
                    'success': False,
                    'symbol': self.symbol,
                    'message': f"Download stopped by user"
                })
                return

            # Build summary
            succeeded = sum(1 for v in tf_results.values() if v == 'OK')
            failed = total_tf - succeeded
            tf_summary = ', '.join(f"{tf}:{status}" for tf, status in tf_results.items())

            if succeeded > 0:
                self.progress.emit(100, 100, f"Done! {succeeded}/{total_tf} timeframes saved")
                self.result.emit({
                    'success': True,
                    'symbol': self.symbol,
                    'message': f"Downloaded {self.max_candles:,} candles x {succeeded} timeframes for {self.symbol} [{tf_summary}]"
                })
            else:
                self.result.emit({
                    'success': False,
                    'symbol': self.symbol,
                    'message': f"Failed all timeframes for {self.symbol} [{tf_summary}]"
                })

        except Exception as e:
            self.error.emit(str(e))
            self.result.emit({
                'success': False,
                'symbol': self.symbol,
                'message': str(e)
            })
        finally:
            self.finished.emit()

    def stop(self):
        """Request stop"""
        self._stopped = True


class TrainingWorker(QThread):
    """
    Worker for model training.

    Emits:
        progress: (current, total, message) during training
        epoch_complete: (epoch, train_loss, val_loss, metrics) per epoch
        result: Training results dict
        error: Error message string
        finished: When work completes
    """
    progress = pyqtSignal(int, int, str)
    epoch_complete = pyqtSignal(int, float, float, dict)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: dict, symbols: List[str]):
        super().__init__()
        self.config = config
        self.symbols = symbols
        self._stopped = False

    def run(self):
        try:
            self.progress.emit(0, 100, "Initializing training...")

            # Import RankNet model
            from src.ml.models.ranknet_model import RankNetPredictor
            from src.core.config import AppConfig
            import numpy as np
            from pathlib import Path

            # Results storage
            all_results = {}
            total_symbols = len(self.symbols)

            # Train ONE MODEL PER COIN
            for sym_idx, symbol in enumerate(self.symbols):
                if self._stopped:
                    return

                base_progress = int((sym_idx / total_symbols) * 90)
                self.progress.emit(base_progress, 100, f"Training {symbol} ({sym_idx+1}/{total_symbols})...")

                try:
                    # Load data for this symbol
                    dataset_dir = Path(AppConfig.DATASET_DIR)
                    symbol_dirs = list(dataset_dir.glob(f"{symbol}/*"))

                    if not symbol_dirs:
                        print(f"[TRAIN] Skipping {symbol}: no data found")
                        continue

                    # Find largest dataset
                    csv_files = []
                    for d in symbol_dirs:
                        csv_files.extend(list(d.glob("*.csv")))

                    if not csv_files:
                        print(f"[TRAIN] Skipping {symbol}: no CSV files")
                        continue

                    # Use largest file
                    largest_file = max(csv_files, key=lambda f: f.stat().st_size)
                    df = pd.read_csv(largest_file)

                    if len(df) < 1000:
                        print(f"[TRAIN] Skipping {symbol}: insufficient data ({len(df)} rows)")
                        continue

                    # Determine timeframe from dataset metadata
                    import json as _json
                    data_timeframe = '1h'  # default
                    metadata_path = largest_file.parent / 'metadata.json'
                    if metadata_path.exists():
                        try:
                            with open(metadata_path) as mf:
                                meta = _json.load(mf)
                            data_timeframe = meta.get('interval', '1h')
                        except Exception:
                            pass

                    print(f"[TRAIN] {symbol}: Loaded {len(df)} rows from {largest_file.name} (timeframe: {data_timeframe})")

                    # Ensure OHLCV columns
                    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing = [c for c in ohlcv_cols if c not in df.columns]
                    if missing:
                        print(f"[TRAIN] Skipping {symbol}: missing columns {missing}")
                        continue

                    # Create RankNet predictor and train on full data
                    # RankNet uses full-batch training (no mini-batching):
                    # ranknet_loss() samples 1024 random pairs from ALL data,
                    # so the full dataset must be passed each epoch.
                    predictor = RankNetPredictor(
                        hidden_dim=int(self.config.get('hidden_size', 128)),
                        num_blocks=int(self.config.get('num_blocks', 3)),
                        dropout=float(self.config.get('dropout', 0.1)),
                        epochs=int(self.config.get('max_epochs', 20)),
                        lr=float(self.config.get('learning_rate', 0.001)),
                        future_horizon=30,
                        take_profit=0.03,
                        stop_loss=0.0075,
                        confidence_threshold=0.9,
                        timeframe=data_timeframe,
                    )

                    self.progress.emit(base_progress + 2, 100, f"Training RankNet for {symbol} ({data_timeframe})...")
                    train_metrics = predictor.train(df[ohlcv_cols], verbose=True)

                    # Save model with timeframe-specific name
                    model_path = f"models/checkpoints/{symbol}/ranknet_{data_timeframe}.pt"
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    predictor.save(model_path)

                    all_results[symbol] = {
                        'model_path': model_path,
                        'accuracy': train_metrics['directional_accuracy'],
                        'ic': train_metrics['ic'],
                        'sharpe_ratio': train_metrics['sharpe'],
                        'edge': train_metrics['directional_accuracy'] - 0.5,
                        'backtest_trades': train_metrics['backtest_trades'],
                        'backtest_win_rate': train_metrics['backtest_win_rate'],
                    }

                    # Emit results for this symbol
                    self.epoch_complete.emit(
                        sym_idx + 1,
                        train_metrics['sharpe'],
                        train_metrics['ic'],
                        {
                            'symbol': symbol,
                            'accuracy': train_metrics['directional_accuracy'],
                            'ic': train_metrics['ic'],
                            'sharpe': train_metrics['sharpe'],
                            'edge_pct': (train_metrics['directional_accuracy'] - 0.5) * 100
                        }
                    )

                except Exception as e:
                    print(f"[TRAIN] Error training {symbol}: {e}")
                    traceback.print_exc()
                    all_results[symbol] = {'error': str(e)}
                    continue

            if self._stopped:
                return

            self.progress.emit(95, 100, "Generating summary...")

            # Summary statistics
            successful = {k: v for k, v in all_results.items() if 'error' not in v}
            avg_sharpe = np.mean([v['sharpe_ratio'] for v in successful.values()]) if successful else 0
            avg_ic = np.mean([v['ic'] for v in successful.values()]) if successful else 0
            avg_acc = np.mean([v['accuracy'] for v in successful.values()]) if successful else 0

            self.progress.emit(100, 100, "Training complete!")
            self.result.emit({
                'success': True,
                'models_trained': len(successful),
                'models_failed': len(all_results) - len(successful),
                'avg_accuracy': avg_acc,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_ic': avg_ic,
                'results': all_results,
                'message': f"Trained {len(successful)}/{len(self.symbols)} RankNet models. Avg DirAcc: {avg_acc:.1%}, Sharpe: {avg_sharpe:.2f}, IC: {avg_ic:.4f}"
            })

        except StopIteration:
            self.result.emit({
                'success': False,
                'message': "Training stopped by user"
            })
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.error.emit(error_msg)
            self.result.emit({'success': False, 'message': error_msg})
        finally:
            self.finished.emit()

    def stop(self):
        """Request stop"""
        self._stopped = True


class PredictionWorker(QThread):
    """
    Worker for running model predictions or tests.

    Emits:
        result: Prediction/test results dict
        error: Error message string
        finished: When work completes
    """
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_path: str, symbol: str, api_client=None, test_mode: bool = False):
        super().__init__()
        self.model_path = model_path
        self.symbol = symbol
        self.api_client = api_client
        self.test_mode = test_mode

    def run(self):
        try:
            from src.ml.inference.predictor import CryptoPredictor
            import numpy as np

            # Find scaler path
            scaler_path = "models/checkpoints/scalers.pkl"

            predictor = CryptoPredictor(
                self.model_path,
                scaler_path=scaler_path,
                verbose=False
            )

            if self.test_mode:
                # Run model test/evaluation
                test_results = predictor.evaluate(self.symbol)
                self.result.emit({
                    'success': True,
                    'symbol': self.symbol,
                    **test_results
                })
            else:
                # Run prediction
                predictions = predictor.predict(self.symbol)

                # Extract predictions (predictor returns 'median' key)
                prices = predictions.get('median', np.array([]))
                if isinstance(prices, np.ndarray):
                    prices = prices.tolist()

                # Calculate confidence from upper/lower bounds
                upper = predictions.get('upper_80', np.array([]))
                lower = predictions.get('lower_80', np.array([]))
                if len(upper) > 0 and len(lower) > 0:
                    if isinstance(upper, np.ndarray):
                        upper = upper.tolist()
                    if isinstance(lower, np.ndarray):
                        lower = lower.tolist()
                    confidence = [(u - l) / 2 for u, l in zip(upper, lower)]
                else:
                    confidence = [0] * len(prices)

                # Get current price (first prediction approximates current)
                current_price = prices[0] if len(prices) > 0 else 0
                predicted_price = prices[-1] if len(prices) > 0 else 0

                # Calculate price change percentage
                price_change_pct = 0.0
                if current_price > 0:
                    price_change_pct = ((predicted_price - current_price) / current_price) * 100

                # Determine trading signal
                if price_change_pct >= 2.0:
                    signal = "STRONG_BUY"
                elif price_change_pct >= 0.5:
                    signal = "BUY"
                elif price_change_pct <= -2.0:
                    signal = "STRONG_SELL"
                elif price_change_pct <= -0.5:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                self.result.emit({
                    'success': True,
                    'symbol': self.symbol,
                    'predictions': prices,
                    'confidence': confidence,
                    'signal': signal,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'price_change_pct': price_change_pct
                })

        except Exception as e:
            error_msg = f"{'Test' if self.test_mode else 'Prediction'} failed: {str(e)}"
            self.error.emit(error_msg)
            self.result.emit({'success': False, 'error': error_msg})
        finally:
            self.finished.emit()


class BacktestWorker(QThread):
    """
    Worker for running backtests.

    Supports both model-based and strategy-based backtesting.

    Emits:
        progress: (current, total, message) during backtest
        result: BacktestResults object
        error: Error message string
        finished: When work completes
    """
    progress = pyqtSignal(int, int, str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: dict, api_client=None, data_manager=None):
        super().__init__()
        self.config = config
        self.api_client = api_client
        self.data_manager = data_manager
        self._stopped = False

    def run(self):
        try:
            from src.ml.backtest.backtester import CryptoBacktester, BacktestConfig
            from src.core.config import AppConfig

            mode = self.config.get('mode', 'strategy')
            symbol = self.config.get('symbol', 'BTCUSDT')

            self.progress.emit(10, 100, "Initializing backtester...")

            # Create backtester
            backtest_config = BacktestConfig(
                initial_capital=float(self.config.get('initial_capital', 1000.0)),
                trade_fee=float(self.config.get('trade_fee', 0.001)),
                confidence_threshold=float(self.config.get('confidence_threshold', 0.5))
            )
            backtester = CryptoBacktester(config=backtest_config, verbose=False)

            if mode == 'strategy':
                # Strategy-based backtest
                strategy_name = self.config.get('strategy', 'rsi')

                self.progress.emit(20, 100, f"Loading data for {symbol}...")

                # Load data
                from src.ml.preprocessing.preprocessor import CryptoPreprocessor
                preprocessor = CryptoPreprocessor(AppConfig.DATASET_DIR)

                # Try to load from saved dataset first
                try:
                    df = preprocessor.load_symbol_data(symbol)
                except:
                    # Fall back to fetching new data if available
                    if self.api_client:
                        import pandas as pd
                        klines = self.api_client.get_klines_formatted(symbol, "1h", 5000)
                        df = pd.DataFrame(klines)
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df.set_index('datetime', inplace=True)
                    else:
                        raise ValueError(f"No data available for {symbol}")

                if self._stopped:
                    return

                # Generate indicators
                self.progress.emit(40, 100, "Generating indicators...")
                df = preprocessor.generate_technical_indicators(df)
                df = df.dropna()

                if len(df) < 100:
                    raise ValueError(f"Insufficient data for {symbol}: only {len(df)} rows after indicator calculation")

                if self._stopped:
                    return

                # Get strategy
                self.progress.emit(60, 100, f"Running {strategy_name} strategy...")

                from src.ml.strategies import get_strategy
                strategy = get_strategy(strategy_name)

                # Run backtest
                results = backtester.run_backtest_with_strategy(strategy, df)

                self.progress.emit(100, 100, "Backtest complete!")
                self.result.emit(results)

            elif mode == 'walk_forward':
                # Walk-forward backtesting using RankNet
                from src.ml.models.ranknet_model import RankNetPredictor
                from pathlib import Path
                import numpy as np
                import pandas as pd
                from dataclasses import dataclass

                n_folds = int(self.config.get('n_folds', 5))
                train_ratio = float(self.config.get('train_ratio', 0.7))
                gap_periods = int(self.config.get('gap_periods', 4))
                retrain = self.config.get('retrain', True)
                model_path = self.config.get('model_path', '')

                self.progress.emit(20, 100, f"Loading data for {symbol}...")

                # Load data
                from pathlib import Path
                dataset_dir = Path(AppConfig.DATASET_DIR)
                symbol_dirs = list(dataset_dir.glob(f"{symbol}/*"))

                if not symbol_dirs:
                    raise ValueError(f"No data found for {symbol}")

                # Find largest dataset
                csv_files = []
                for d in symbol_dirs:
                    csv_files.extend(list(d.glob("*.csv")))

                if not csv_files:
                    raise ValueError(f"No CSV files found for {symbol}")

                largest_file = max(csv_files, key=lambda f: f.stat().st_size)
                df = pd.read_csv(largest_file)

                if len(df) < 1000:
                    raise ValueError(f"Insufficient data for walk-forward: {len(df)} rows")

                # Ensure OHLCV columns
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                missing = [c for c in ohlcv_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"Missing OHLCV columns: {missing}")

                if self._stopped:
                    return

                # Calculate fold sizes
                fold_size = len(df) // n_folds
                train_size = int(fold_size * train_ratio)
                test_size = fold_size - train_size - gap_periods

                if test_size < 100:
                    raise ValueError(f"Test fold too small ({test_size} rows). Reduce folds or increase data.")

                self.progress.emit(40, 100, f"Running {n_folds}-fold walk-forward backtest...")

                # Walk-forward results storage
                fold_results = []
                equity_values = [backtest_config.initial_capital]
                current_capital = backtest_config.initial_capital
                predictor = None

                for fold_idx in range(n_folds):
                    if self._stopped:
                        return

                    fold_start = fold_idx * fold_size
                    train_end = fold_start + train_size
                    test_start = train_end + gap_periods
                    test_end = min(test_start + test_size, len(df))

                    if test_end > len(df):
                        break

                    train_df = df.iloc[fold_start:train_end][ohlcv_cols].copy()
                    test_df = df.iloc[test_start:test_end][ohlcv_cols].copy()

                    self.progress.emit(
                        40 + int((fold_idx / n_folds) * 50),
                        100,
                        f"Fold {fold_idx + 1}/{n_folds}: Training RankNet..."
                    )

                    # Train or reuse model
                    if retrain or fold_idx == 0 or predictor is None:
                        predictor = RankNetPredictor(
                            hidden_dim=int(self.config.get('hidden_size', 128)),
                            num_blocks=int(self.config.get('num_blocks', 3)),
                            dropout=float(self.config.get('dropout', 0.1)),
                            epochs=int(self.config.get('max_epochs', 20)),
                            lr=float(self.config.get('learning_rate', 0.001)),
                            future_horizon=30,
                            take_profit=0.03, stop_loss=0.0075,
                            confidence_threshold=0.9,
                        )
                        if len(train_df) >= 500:
                            train_metrics = predictor.train(train_df, verbose=False)
                        else:
                            print(f"  Fold {fold_idx + 1}: Skipped (train too small: {len(train_df)})")
                            continue
                    elif model_path and Path(model_path).exists():
                        predictor = RankNetPredictor.load(model_path)

                    # Evaluate on test fold
                    if len(test_df) >= 300:
                        test_result = predictor.predict(test_df)
                        test_metrics = test_result['metrics']
                    else:
                        print(f"  Fold {fold_idx + 1}: Test too small ({len(test_df)})")
                        continue

                    # Track metrics
                    fold_metrics = {
                        'fold': fold_idx + 1,
                        'accuracy': test_metrics.get('directional_accuracy', 0.5),
                        'ic': test_metrics.get('ic', 0),
                        'sharpe': test_metrics.get('sharpe', 0),
                        'edge': test_metrics.get('directional_accuracy', 0.5) - 0.5,
                        'n_samples': test_metrics.get('trades', 0)
                    }
                    fold_results.append(fold_metrics)

                    # Use actual backtest results from RankNet
                    win_rate = test_metrics.get('win_rate', 0.5)
                    n_trades = test_metrics.get('trades', 0)

                    avg_win = 0.03   # TP target
                    avg_loss = 0.0075  # SL target

                    for _ in range(min(n_trades, 100)):
                        if np.random.random() < win_rate:
                            current_capital *= (1 + avg_win * 0.1 - backtest_config.trade_fee)
                        else:
                            current_capital *= (1 - avg_loss * 0.1 - backtest_config.trade_fee)
                        equity_values.append(current_capital)

                    print(f"  Fold {fold_idx + 1}: DirAcc={fold_metrics['accuracy']:.1%}, "
                          f"IC={fold_metrics['ic']:.4f}, Sharpe={fold_metrics['sharpe']:.2f}")

                # Aggregate results
                avg_accuracy = np.mean([f['accuracy'] for f in fold_results])
                avg_ic = np.mean([f['ic'] for f in fold_results])
                avg_sharpe = np.mean([f['sharpe'] for f in fold_results])
                sharpe_std = np.std([f['sharpe'] for f in fold_results])

                # Stability assessment
                sharpe_cv = sharpe_std / abs(avg_sharpe) if avg_sharpe != 0 else float('inf')
                if sharpe_cv < 0.5:
                    stability = "Stable"
                elif sharpe_cv < 1.0:
                    stability = "Moderate"
                else:
                    stability = "Unstable"

                # Create walk-forward results object
                @dataclass
                class WalkForwardResults:
                    n_folds: int
                    fold_results: list
                    avg_accuracy: float
                    avg_ic: float
                    avg_sharpe: float
                    stability: str
                    initial_capital: float
                    final_capital: float
                    equity_curve: object  # pandas Series

                import pandas as pd
                equity_series = pd.Series(equity_values)

                results = WalkForwardResults(
                    n_folds=len(fold_results),
                    fold_results=fold_results,
                    avg_accuracy=avg_accuracy,
                    avg_ic=avg_ic,
                    avg_sharpe=avg_sharpe,
                    stability=stability,
                    initial_capital=backtest_config.initial_capital,
                    final_capital=current_capital,
                    equity_curve=equity_series
                )

                self.progress.emit(100, 100, "Walk-forward backtest complete!")
                self.result.emit(results)

            else:
                # Model-based backtest (Directionality Model / LSTM)
                model_path = self.config.get('model_path')
                if not model_path:
                    raise ValueError("model_path required for model-based backtest")

                self.progress.emit(20, 100, "Loading model...")

                import torch
                import numpy as np
                import pandas as pd
                from pathlib import Path

                # Detect model type and load appropriately
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Check if saved model is RankNet (has hidden_dim in config)
                is_ranknet = (isinstance(checkpoint, dict)
                              and 'config' in checkpoint
                              and isinstance(checkpoint['config'], dict)
                              and 'hidden_dim' in checkpoint.get('config', {}))

                if is_ranknet:
                    # Load RankNet model
                    self.progress.emit(25, 100, "Loading RankNet model...")
                    from src.ml.models.ranknet_model import RankNetPredictor

                    predictor = RankNetPredictor.load(model_path)
                    print(f"[BACKTEST] RankNet model loaded: {len(predictor.feature_cols)} features")

                    self.progress.emit(40, 100, f"Loading data for {symbol}...")

                    # Load data
                    dataset_dir = Path(AppConfig.DATASET_DIR)
                    symbol_dirs = list(dataset_dir.glob(f"{symbol}/*"))

                    if not symbol_dirs:
                        raise ValueError(f"No data found for {symbol}")

                    csv_files = []
                    for d in symbol_dirs:
                        csv_files.extend(list(d.glob("*.csv")))

                    if not csv_files:
                        raise ValueError(f"No CSV files found for {symbol}")

                    largest_file = max(csv_files, key=lambda f: f.stat().st_size)
                    df = pd.read_csv(largest_file)
                    print(f"[BACKTEST] Loaded {len(df)} rows from {largest_file.name}")

                    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing = [c for c in ohlcv_cols if c not in df.columns]
                    if missing:
                        raise ValueError(f"Missing OHLCV columns: {missing}")

                    if len(df) < 500:
                        raise ValueError(f"Insufficient data for backtest: {len(df)} rows")

                    if self._stopped:
                        return

                    self.progress.emit(60, 100, "Running RankNet predictions & backtest...")

                    # Use RankNet's built-in pipeline (exact ver11_gpt processing)
                    result = predictor.predict(df[ohlcv_cols])
                    predictions = result['predictions']
                    metrics = result['metrics']

                    print(f"[BACKTEST] RankNet predictions: {len(predictions)}, "
                          f"IC={metrics['ic']:.4f}, DirAcc={metrics['directional_accuracy']:.1%}")

                    # Prepare data for backtester display
                    from src.ml.models.ranknet_model import (
                        apply_triple_barrier, compute_regime_features,
                        regime_filter, generate_feature_matrix,
                        compute_future_return
                    )
                    df_proc = df[ohlcv_cols].copy()
                    df_proc = apply_triple_barrier(df_proc)
                    df_proc = compute_regime_features(df_proc)
                    df_proc = regime_filter(df_proc)
                    X_tmp, _ = generate_feature_matrix(df_proc)
                    df_proc = df_proc.loc[X_tmp.index]
                    df_proc = compute_future_return(df_proc, horizon=predictor.future_horizon)

                    actuals = df_proc['future_return'].values[:len(predictions)]
                    actuals = np.nan_to_num(actuals, nan=0.0)
                    prices = df_proc['close'].values[:len(predictions)]

                    if 'datetime' in df.columns:
                        timestamps = pd.to_datetime(df_proc['datetime'].iloc[:len(predictions)]).reset_index(drop=True)
                    elif 'timestamp' in df.columns:
                        ts_vals = df.loc[df_proc.index[:len(predictions)], 'timestamp']
                        timestamps = pd.to_datetime(ts_vals, unit='ms').reset_index(drop=True)
                    else:
                        timestamps = pd.date_range(start='2024-01-01', periods=len(predictions), freq='h')

                    self.progress.emit(90, 100, "Running backtest...")

                    min_len = min(len(predictions), len(actuals), len(prices), len(timestamps))
                    predictions = predictions[:min_len]
                    actuals = actuals[:min_len]
                    prices = prices[:min_len]
                    timestamps = timestamps[:min_len]

                    try:
                        results = backtester.run_backtest_from_returns(
                            predicted_returns=predictions,
                            actual_returns=actuals,
                            current_prices=prices,
                            timestamps=timestamps,
                            denormalize=False
                        )
                    except Exception as backtest_error:
                        print(f"[BACKTEST] run_backtest_from_returns failed: {type(backtest_error).__name__}: {backtest_error}")
                        import traceback
                        traceback.print_exc()
                        raise

                else:
                    # Directionality / legacy model: train RankNet on-the-fly
                    # and backtest using ver11_gpt-style (fixed TP/SL + confidence)
                    self.progress.emit(25, 100, "Preparing on-the-fly RankNet backtest...")
                    from src.ml.models.ranknet_model import RankNetPredictor

                    self.progress.emit(40, 100, f"Loading data for {symbol}...")

                    dataset_dir = Path(AppConfig.DATASET_DIR)
                    symbol_dirs = list(dataset_dir.glob(f"{symbol}/*"))

                    if not symbol_dirs:
                        raise ValueError(f"No data found for {symbol}")

                    csv_files = []
                    for d in symbol_dirs:
                        csv_files.extend(list(d.glob("*.csv")))

                    if not csv_files:
                        raise ValueError(f"No CSV files found for {symbol}")

                    largest_file = max(csv_files, key=lambda f: f.stat().st_size)
                    df = pd.read_csv(largest_file)
                    print(f"[BACKTEST] Loaded {len(df)} rows from {largest_file.name}")

                    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing = [c for c in ohlcv_cols if c not in df.columns]
                    if missing:
                        raise ValueError(f"Missing OHLCV columns: {missing}")

                    if len(df) < 500:
                        raise ValueError(f"Insufficient data for backtest: {len(df)} rows")

                    if self._stopped:
                        return

                    self.progress.emit(50, 100, "Training RankNet on-the-fly...")

                    predictor = RankNetPredictor(
                        hidden_dim=int(self.config.get('hidden_size', 128)),
                        num_blocks=int(self.config.get('num_blocks', 3)),
                        dropout=float(self.config.get('dropout', 0.1)),
                        epochs=int(self.config.get('max_epochs', 20)),
                        lr=float(self.config.get('learning_rate', 0.001)),
                        future_horizon=30,
                        take_profit=0.03, stop_loss=0.0075,
                        confidence_threshold=0.9,
                    )

                    self.progress.emit(60, 100, "Training & evaluating...")
                    result_rn = predictor.predict_direction_only(df[ohlcv_cols])
                    metrics = result_rn['metrics']

                    # Convert ver11_gpt metrics to BacktestResults-compatible object
                    from dataclasses import dataclass as _dc, field as _field

                    @_dc
                    class OnTheFlyBacktestResults:
                        initial_capital: float
                        final_capital: float
                        total_return: float
                        total_return_pct: float
                        num_trades: int
                        win_rate: float
                        profit_loss_ratio: float
                        buy_hold_return: float = 0.0
                        buy_hold_return_pct: float = 0.0
                        excess_return: float = 0.0
                        direction_accuracy: float = 0.0
                        equity_curve: object = _field(default_factory=lambda: pd.Series(dtype=float))
                        sharpe_ratio: float = 0.0
                        max_drawdown: float = 0.0
                        information_coefficient: float = 0.0
                        num_wins: int = 0
                        num_losses: int = 0
                        prediction_mae: float = 0.0
                        prediction_rmse: float = 0.0
                        prediction_r2: float = 0.0
                        sortino_ratio: float = 0.0
                        max_drawdown_pct: float = 0.0
                        calmar_ratio: float = 0.0
                        volatility: float = 0.0
                        trades: list = _field(default_factory=list)

                    init_cap = metrics.get('initial_capital', 10000)
                    final_cap = metrics.get('final_capital', init_cap)
                    total_ret = final_cap - init_cap
                    total_ret_pct = (total_ret / init_cap * 100) if init_cap > 0 else 0
                    n_trades = metrics.get('trades', 0)
                    wr = metrics.get('win_rate', 0)

                    # Buy-and-hold comparison
                    prices = df['close'].values
                    bh_ret_pct = ((prices[-1] - prices[0]) / prices[0] * 100) if len(prices) > 1 else 0

                    results = OnTheFlyBacktestResults(
                        initial_capital=init_cap,
                        final_capital=final_cap,
                        total_return=total_ret,
                        total_return_pct=total_ret_pct,
                        num_trades=n_trades,
                        num_wins=int(wr * n_trades),
                        num_losses=n_trades - int(wr * n_trades),
                        win_rate=wr,
                        profit_loss_ratio=metrics.get('sharpe', 0),
                        buy_hold_return=bh_ret_pct * init_cap / 100,
                        buy_hold_return_pct=bh_ret_pct,
                        excess_return=total_ret_pct - bh_ret_pct,
                        direction_accuracy=metrics.get('directional_accuracy', 0.5),
                        sharpe_ratio=metrics.get('sharpe', 0),
                        information_coefficient=metrics.get('ic', 0),
                        equity_curve=pd.Series(dtype=float),
                    )

                    self.progress.emit(100, 100, "Backtest complete!")
                    self.result.emit(results)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[BACKTEST] ERROR: {error_details}")
            error_msg = f"Backtest failed: {str(e)}" if str(e) else f"Backtest failed: {type(e).__name__}"
            if not error_msg or error_msg == "Backtest failed: ":
                error_msg = f"Backtest failed: {type(e).__name__}\n{error_details[-500:]}"
            self.error.emit(error_msg)
        finally:
            self.finished.emit()

    def stop(self):
        """Request stop"""
        self._stopped = True


class SignalScannerWorker(QThread):
    """
    Worker for scanning multiple cryptocurrencies for trading signals.

    Scans symbols for buy/sell signals based on selected combo strategies
    within a specified timeframe, with AND/OR logic support.

    Emits:
        progress: (current, total, message) during scanning
        result: List of signal results
        error: Error message string
        finished: When work completes
    """
    progress = pyqtSignal(int, int, str)
    result = pyqtSignal(list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: dict, api_client, market_data: list):
        super().__init__()
        self.config = config
        self.api_client = api_client
        self.market_data = market_data
        self._stopped = False

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        multipliers = {
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800,
        }

        # Extract number and unit (e.g., '1h' -> 1, 'h')
        unit = timeframe[-1].lower()
        try:
            value = int(timeframe[:-1])
        except ValueError:
            value = 1

        return value * multipliers.get(unit, 3600)  # Default to 1 hour

    def run(self):
        try:
            import pandas as pd
            import numpy as np
            from datetime import datetime, timezone, timedelta
            from src.api.ccxt_client import get_ccxt_client

            timeframe = self.config.get('timeframe', '1h')
            past_candles = self.config.get('past_candles', 5)
            strategies = self.config.get('strategies', ['rsi_macd'])
            logic_mode = self.config.get('logic_mode', 'or')  # 'and' or 'or'
            include_buy = self.config.get('include_buy', True)
            include_sell = self.config.get('include_sell', True)

            # Calculate the maximum age for signals based on timeframe and past_candles
            timeframe_seconds = self._timeframe_to_seconds(timeframe)
            max_signal_age_seconds = past_candles * timeframe_seconds
            # Add a small buffer (1 extra candle) for edge cases
            max_signal_age_seconds += timeframe_seconds
            cutoff_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=max_signal_age_seconds)

            # Use all market symbols (no volume filter)
            filtered_symbols = self.market_data[:100]  # Limit to top 100 by volume

            if not filtered_symbols:
                self.progress.emit(100, 100, "No symbols to scan")
                self.result.emit([])
                return

            self.progress.emit(0, len(filtered_symbols), f"Scanning {len(filtered_symbols)} symbols on {timeframe}...")

            # Import strategies
            from src.ml.strategies import get_strategy

            # Get CCXT client for timeframe-specific data
            ccxt_client = get_ccxt_client('binance')

            all_signals = []
            candles_needed = max(200, past_candles + 100)  # Extra for indicator warmup

            for idx, symbol_data in enumerate(filtered_symbols):
                if self._stopped:
                    break

                symbol = symbol_data.get('symbol', '')
                self.progress.emit(idx + 1, len(filtered_symbols), f"Scanning {symbol} ({timeframe})...")

                try:
                    # Convert symbol format for CCXT
                    ccxt_symbol = ccxt_client.convert_symbol_format(symbol, to_ccxt=True)

                    # Fetch candles at specified timeframe
                    df = ccxt_client.get_ohlcv(ccxt_symbol, timeframe, candles_needed)

                    if df is None or len(df) < 50:
                        continue

                    # Calculate indicators needed for strategies
                    df = self._calculate_indicators(df)
                    df['datetime'] = df.index  # Add datetime as column for strategies

                    # Collect signals from each strategy for this symbol
                    symbol_strategy_signals = {}  # strategy_name -> list of recent signals

                    for strategy_name in strategies:
                        if self._stopped:
                            break

                        try:
                            strategy = get_strategy(strategy_name)

                            # Check if data has required features
                            is_valid, missing = strategy.validate_data(df)
                            if not is_valid:
                                continue

                            # Generate signals
                            signals = strategy.generate_signals(df)

                            # Get signals from last N candles only
                            recent_signals = signals[-past_candles:] if len(signals) >= past_candles else signals

                            # Filter to BUY/SELL with confidence > 0 and within time window
                            valid_signals = []
                            for signal in recent_signals:
                                if signal.action == 'HOLD':
                                    continue
                                if signal.action == 'BUY' and not include_buy:
                                    continue
                                if signal.action == 'SELL' and not include_sell:
                                    continue
                                if signal.confidence <= 0:
                                    continue

                                # Time-based filter: only include signals within the allowed time window
                                if signal.timestamp is not None:
                                    try:
                                        # Convert signal timestamp to datetime for comparison
                                        if isinstance(signal.timestamp, pd.Timestamp):
                                            signal_time = signal.timestamp.to_pydatetime()
                                            if signal_time.tzinfo is not None:
                                                signal_time = signal_time.replace(tzinfo=None)
                                        elif isinstance(signal.timestamp, datetime):
                                            signal_time = signal.timestamp
                                            if signal_time.tzinfo is not None:
                                                signal_time = signal_time.replace(tzinfo=None)
                                        else:
                                            signal_time = pd.to_datetime(signal.timestamp).to_pydatetime()

                                        # Skip signals older than the cutoff time
                                        if signal_time < cutoff_time:
                                            continue
                                    except Exception:
                                        pass  # If timestamp conversion fails, include the signal

                                valid_signals.append(signal)

                            symbol_strategy_signals[strategy_name] = valid_signals

                        except Exception as strategy_error:
                            continue

                    # Apply AND/OR logic
                    if logic_mode == 'and':
                        # AND mode: All selected strategies must have matching signals
                        result_signals = self._apply_and_logic(
                            symbol, symbol_data, symbol_strategy_signals, strategies
                        )
                    else:
                        # OR mode: Any strategy signal is included
                        result_signals = self._apply_or_logic(
                            symbol, symbol_data, symbol_strategy_signals
                        )

                    all_signals.extend(result_signals)

                except Exception as symbol_error:
                    continue

            # Sort by confidence (highest first)
            all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)

            # Remove duplicates (same symbol + signal type, keep highest confidence)
            seen = set()
            unique_signals = []
            for sig in all_signals:
                key = (sig['symbol'], sig['signal'])
                if key not in seen:
                    seen.add(key)
                    unique_signals.append(sig)

            self.progress.emit(len(filtered_symbols), len(filtered_symbols), "Scan complete!")
            self.result.emit(unique_signals)

        except Exception as e:
            error_msg = f"Signal scan failed: {str(e)}"
            self.error.emit(error_msg)
            import traceback
            traceback.print_exc()
        finally:
            self.finished.emit()

    def _apply_or_logic(self, symbol: str, symbol_data: dict,
                        strategy_signals: dict) -> list:
        """OR logic: Include any signal from any strategy"""
        results = []

        for strategy_name, signals in strategy_signals.items():
            for signal in signals:
                results.append({
                    'symbol': symbol,
                    'signal': signal.action,
                    'strategy': self._get_strategy_display_name(strategy_name),
                    'confidence': signal.confidence,
                    'price': signal.price if signal.price > 0 else symbol_data.get('price', 0),
                    'change_pct': symbol_data.get('price_change_pct', 0),
                    'timestamp': signal.timestamp,
                })

        return results

    def _apply_and_logic(self, symbol: str, symbol_data: dict,
                         strategy_signals: dict, selected_strategies: list) -> list:
        """AND logic: Only include if ALL strategies agree on direction"""
        results = []

        # Check if all selected strategies have signals
        strategies_with_signals = set(strategy_signals.keys())
        if not strategies_with_signals.issuperset(set(selected_strategies)):
            return results  # Not all strategies have signals

        # Collect all BUY and SELL votes
        buy_count = 0
        sell_count = 0
        total_buy_confidence = 0.0
        total_sell_confidence = 0.0
        latest_timestamp = None

        for strategy_name in selected_strategies:
            signals = strategy_signals.get(strategy_name, [])
            for signal in signals:
                if signal.action == 'BUY':
                    buy_count += 1
                    total_buy_confidence += signal.confidence
                elif signal.action == 'SELL':
                    sell_count += 1
                    total_sell_confidence += signal.confidence

                if signal.timestamp is not None:
                    if latest_timestamp is None or signal.timestamp > latest_timestamp:
                        latest_timestamp = signal.timestamp

        # Only emit signal if ALL strategies agree
        num_strategies = len(selected_strategies)

        if buy_count >= num_strategies:
            # All strategies signal BUY
            avg_confidence = total_buy_confidence / buy_count
            results.append({
                'symbol': symbol,
                'signal': 'BUY',
                'strategy': f"ALL ({num_strategies} strategies)",
                'confidence': avg_confidence,
                'price': symbol_data.get('price', 0),
                'change_pct': symbol_data.get('price_change_pct', 0),
                'timestamp': latest_timestamp,
            })
        elif sell_count >= num_strategies:
            # All strategies signal SELL
            avg_confidence = total_sell_confidence / sell_count
            results.append({
                'symbol': symbol,
                'signal': 'SELL',
                'strategy': f"ALL ({num_strategies} strategies)",
                'confidence': avg_confidence,
                'price': symbol_data.get('price', 0),
                'change_pct': symbol_data.get('price_change_pct', 0),
                'timestamp': latest_timestamp,
            })

        return results

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators needed for strategies"""
        import numpy as np

        close = df['close']
        high = df['high']
        low = df['low']

        # Moving Averages
        df['sma_10'] = close.rolling(window=10).mean()
        df['sma_20'] = close.rolling(window=20).mean()
        df['sma_50'] = close.rolling(window=50).mean()
        df['ema_5'] = close.ewm(span=5, adjust=False).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_13'] = close.ewm(span=13, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()

        # MA ratios for MA Crossover strategy
        df['sma10_sma20_ratio'] = (df['sma_10'] / df['sma_20'].replace(0, np.nan)) - 1.0
        df['price_sma10_ratio'] = (close / df['sma_10'].replace(0, np.nan)) - 1.0
        df['price_sma20_ratio'] = (close / df['sma_20'].replace(0, np.nan)) - 1.0

        # Bollinger Bands
        df['bb_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        df['bb_position'] = ((close - df['bb_lower']) / bb_range) * 2 - 1
        df['bb_width'] = bb_range / df['bb_middle'].replace(0, np.nan)

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Stochastic
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        stoch_range = (highest_high - lowest_low).replace(0, np.nan)
        df['stoch_k'] = 100 * (close - lowest_low) / stoch_range
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # ADX (Average Directional Index)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.rolling(window=14).mean()

        return df.dropna()

    def _get_strategy_display_name(self, strategy_key: str) -> str:
        """Get display name for strategy"""
        names = {
            'rsi_macd': 'RSI+MACD',
            'bb_rsi': 'BB+RSI',
            'macd_ma': 'MACD+MA',
            'stoch_rsi': 'Stoch+RSI',
            'triple_ema': 'Triple EMA',
            'adx_macd': 'ADX+MACD',
            'stochastic': 'Stochastic',
            'ensemble': 'Ensemble',
        }
        return names.get(strategy_key, strategy_key)

    def stop(self):
        """Request stop"""
        self._stopped = True


class TrainAllWorker(QThread):
    """
    Worker for training directionality models for all cryptos.

    For each crypto:
    - Downloads 50K candles via CCXT
    - If model already exists: loads model, recalculates metrics on recent data, backtests
    - If no model: trains a new RankNet model from scratch
    - Saves model weights, metrics, and backtest results to
      lightning_logs/models/checkpoints/{symbol}/
    - Deletes downloaded data after processing

    Emits:
        progress: (current, total, message) during training
        symbol_complete: dict with per-symbol results
        result: Final summary dict
        error: Error message string
        finished: When work completes
    """
    progress = pyqtSignal(int, int, str)
    symbol_complete = pyqtSignal(dict)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, timeframe: str = '1h', symbols: list = None):
        super().__init__()
        self.timeframe = timeframe
        self.symbols = symbols  # If None, will fetch all USDT symbols
        self._stopped = False

    def run(self):
        try:
            import json
            import numpy as np
            from pathlib import Path
            from datetime import datetime
            from src.api.ccxt_client import get_ccxt_client
            from src.ml.models.ranknet_model import RankNetPredictor

            self.progress.emit(0, 100, "Initializing...")

            # Get symbols
            if self.symbols is None:
                try:
                    from src.gui_pyqt.utils.symbols import get_all_usdt_symbols
                    symbols = get_all_usdt_symbols()
                except Exception:
                    from src.gui_pyqt.utils.symbols import get_default_symbols
                    symbols = get_default_symbols()
            else:
                symbols = self.symbols

            total = len(symbols)
            self.progress.emit(0, total, f"Training {total} symbols on {self.timeframe}...")

            client = get_ccxt_client('binance')
            checkpoint_base = Path("lightning_logs/models/checkpoints")
            all_results = {}
            trained = 0
            skipped = 0
            failed = 0

            for idx, symbol in enumerate(symbols):
                if self._stopped:
                    break

                self.progress.emit(idx, total, f"[{idx+1}/{total}] Processing {symbol}...")

                sym_dir = checkpoint_base / symbol
                model_path = sym_dir / f"ranknet_{self.timeframe}.pt"
                metrics_path = sym_dir / f"metrics_{self.timeframe}.json"
                backtest_path = sym_dir / f"backtest_{self.timeframe}.json"

                try:
                    # Download data
                    ccxt_symbol = client.convert_symbol_format(symbol, to_ccxt=True)
                    self.progress.emit(idx, total, f"[{idx+1}/{total}] Downloading {symbol}...")

                    try:
                        df = client.get_max_ohlcv(ccxt_symbol, self.timeframe, max_candles=50000)
                    except (ValueError, Exception) as e:
                        print(f"[TrainAll] Skipping {symbol}: {e}")
                        skipped += 1
                        continue

                    if df is None or len(df) < 500:
                        print(f"[TrainAll] Skipping {symbol}: insufficient data ({len(df) if df is not None else 0})")
                        skipped += 1
                        continue

                    ohlcv_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

                    if model_path.exists():
                        # Model already trained - recalculate metrics on recent data
                        self.progress.emit(idx, total, f"[{idx+1}/{total}] Re-evaluating {symbol}...")
                        try:
                            predictor = RankNetPredictor.load(str(model_path))
                            result = predictor.predict(ohlcv_df)
                            metrics = result['metrics']
                            direction = result['direction']
                            confidence = result['confidence']
                        except Exception as e:
                            # Model corrupted, retrain
                            print(f"[TrainAll] Model load failed for {symbol}, retraining: {e}")
                            predictor, metrics, direction, confidence = self._train_new(
                                ohlcv_df, symbol, model_path
                            )
                    else:
                        # Train new model
                        self.progress.emit(idx, total, f"[{idx+1}/{total}] Training {symbol}...")
                        predictor, metrics, direction, confidence = self._train_new(
                            ohlcv_df, symbol, model_path
                        )

                    if metrics is None:
                        failed += 1
                        continue

                    # Save metrics
                    sym_dir.mkdir(parents=True, exist_ok=True)
                    metrics_data = {
                        'symbol': symbol,
                        'timeframe': self.timeframe,
                        'timestamp': datetime.utcnow().isoformat(),
                        'direction': direction,
                        'confidence': confidence,
                        'ic': metrics.get('ic', 0),
                        'directional_accuracy': metrics.get('directional_accuracy', 0),
                        'sharpe': metrics.get('sharpe', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'market_coverage': metrics.get('market_coverage', 0),
                        'trades': metrics.get('trades', 0),
                        'equity_generated': metrics.get('equity_generated', 0),
                        'final_capital': metrics.get('final_capital', 10000),
                        'data_rows': len(ohlcv_df),
                    }

                    with open(metrics_path, 'w') as f:
                        json.dump(metrics_data, f, indent=2)

                    # Save backtest results
                    backtest_data = {
                        'symbol': symbol,
                        'timeframe': self.timeframe,
                        'timestamp': datetime.utcnow().isoformat(),
                        'trades': metrics.get('trades', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'sharpe': metrics.get('sharpe', 0),
                        'equity_generated': metrics.get('equity_generated', 0),
                        'final_capital': metrics.get('final_capital', 10000),
                        'initial_capital': metrics.get('initial_capital', 10000),
                    }

                    with open(backtest_path, 'w') as f:
                        json.dump(backtest_data, f, indent=2)

                    trained += 1
                    all_results[symbol] = metrics_data

                    self.symbol_complete.emit(metrics_data)

                    print(f"[TrainAll] {symbol}: {direction} (IC={metrics.get('ic', 0):.4f}, "
                          f"WR={metrics.get('win_rate', 0):.1%})")

                    # Data is in-memory only (DataFrame), no files to delete
                    del df, ohlcv_df

                except Exception as e:
                    print(f"[TrainAll] Error processing {symbol}: {e}")
                    import traceback as tb
                    tb.print_exc()
                    failed += 1
                    continue

            self.progress.emit(total, total, "Training complete!")
            self.result.emit({
                'success': True,
                'trained': trained,
                'skipped': skipped,
                'failed': failed,
                'total': total,
                'timeframe': self.timeframe,
                'results': all_results,
                'message': f"Processed {trained}/{total} symbols ({skipped} skipped, {failed} failed)"
            })

        except Exception as e:
            self.error.emit(f"Train All failed: {str(e)}")
            import traceback as tb
            tb.print_exc()
        finally:
            self.finished.emit()

    def _train_new(self, ohlcv_df, symbol, model_path):
        """Train a new RankNet model and save it."""
        try:
            from src.ml.models.ranknet_model import RankNetPredictor
            from pathlib import Path

            predictor = RankNetPredictor(
                hidden_dim=128, num_blocks=3, dropout=0.1,
                epochs=20, lr=0.001, future_horizon=30,
                take_profit=0.03, stop_loss=0.0075,
                confidence_threshold=0.9, timeframe=self.timeframe,
            )

            predictor.train(ohlcv_df, verbose=False)

            # Save model
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            predictor.save(str(model_path))

            # Get direction from predictions
            result = predictor.predict(ohlcv_df)
            metrics = result['metrics']
            direction = result['direction']
            confidence = result['confidence']

            return predictor, metrics, direction, confidence

        except Exception as e:
            print(f"[TrainAll] Training failed for {symbol}: {e}")
            return None, None, '--', 0.0

    def stop(self):
        """Request stop"""
        self._stopped = True


class AISignalScannerWorker(QThread):
    """
    Worker for scanning cryptos using saved directionality AI models.

    Loads saved RankNet models from lightning_logs/models/checkpoints/,
    downloads recent data for each, runs predictions, and returns
    results filtered by confidence threshold.

    Emits:
        progress: (current, total, message) during scanning
        result: List of signal dicts
        error: Error message string
        finished: When work completes
    """
    progress = pyqtSignal(int, int, str)
    result = pyqtSignal(list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stopped = False

    def run(self):
        try:
            import numpy as np
            from pathlib import Path
            from src.api.ccxt_client import get_ccxt_client
            from src.ml.models.ranknet_model import RankNetPredictor

            confidence_threshold = self.config.get('confidence_threshold', 0.5)
            signal_filter = self.config.get('signal_filter', 'all')  # 'all', 'buy', 'sell'
            timeframe = self.config.get('timeframe', '1h')

            checkpoint_base = Path("lightning_logs/models/checkpoints")

            if not checkpoint_base.exists():
                self.error.emit("No trained models found. Train models first using 'Train All'.")
                return

            # Find all symbol directories with models for this timeframe
            model_dirs = []
            for sym_dir in checkpoint_base.iterdir():
                if sym_dir.is_dir():
                    model_path = sym_dir / f"ranknet_{timeframe}.pt"
                    if model_path.exists():
                        model_dirs.append((sym_dir.name, model_path))

            if not model_dirs:
                self.error.emit(f"No trained models found for timeframe {timeframe}.")
                return

            total = len(model_dirs)
            self.progress.emit(0, total, f"Scanning {total} models...")

            client = get_ccxt_client('binance')
            all_signals = []

            for idx, (symbol, model_path) in enumerate(model_dirs):
                if self._stopped:
                    break

                self.progress.emit(idx, total, f"[{idx+1}/{total}] Evaluating {symbol}...")

                try:
                    # Load model
                    predictor = RankNetPredictor.load(str(model_path))

                    # Download recent data (500 candles for evaluation)
                    ccxt_symbol = client.convert_symbol_format(symbol, to_ccxt=True)
                    try:
                        df = client.get_max_ohlcv(ccxt_symbol, timeframe, max_candles=500)
                    except (ValueError, Exception):
                        continue

                    if df is None or len(df) < 300:
                        continue

                    ohlcv_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

                    # Run prediction
                    pred_result = predictor.predict(ohlcv_df)
                    direction = pred_result['direction']
                    confidence = pred_result['confidence']
                    metrics = pred_result['metrics']

                    # Map direction to signal
                    if direction == 'UP':
                        signal = 'STRONG BUY' if confidence >= 0.7 else 'BUY'
                    elif direction == 'DOWN':
                        signal = 'STRONG SELL' if confidence >= 0.7 else 'SELL'
                    else:
                        signal = 'HOLD'

                    # Apply filters
                    if confidence < confidence_threshold:
                        continue

                    if signal_filter == 'buy' and direction != 'UP':
                        continue
                    if signal_filter == 'sell' and direction != 'DOWN':
                        continue

                    # Get current price info
                    current_price = float(ohlcv_df['close'].iloc[-1])

                    # Compute 24h change from data
                    if len(ohlcv_df) > 24:
                        price_24h_ago = float(ohlcv_df['close'].iloc[-25])
                        change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
                    else:
                        change_24h = 0.0

                    signal_data = {
                        'symbol': symbol,
                        'signal': signal,
                        'direction': direction,
                        'confidence': confidence,
                        'price': current_price,
                        'change_pct': change_24h,
                        'ic': metrics.get('ic', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'sharpe': metrics.get('sharpe', 0),
                        'trades': metrics.get('trades', 0),
                        'equity_generated': metrics.get('equity_generated', 0),
                    }

                    all_signals.append(signal_data)

                    # Free memory
                    del df, ohlcv_df, predictor

                except Exception as e:
                    print(f"[AIScan] Error evaluating {symbol}: {e}")
                    continue

            # Sort by confidence descending
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)

            self.progress.emit(total, total, f"Scan complete: {len(all_signals)} signals")
            self.result.emit(all_signals)

        except Exception as e:
            self.error.emit(f"AI scan failed: {str(e)}")
            import traceback as tb
            tb.print_exc()
        finally:
            self.finished.emit()

    def stop(self):
        """Request stop"""
        self._stopped = True
