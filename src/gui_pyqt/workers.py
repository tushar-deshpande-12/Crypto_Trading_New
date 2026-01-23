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
    Worker for downloading historical data.

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
            self.progress.emit(0, 100, f"Starting download for {self.symbol}...")

            def progress_callback(current, total, message):
                if not self._stopped:
                    self.progress.emit(current, total, message)

            result = self.data_manager.fetch_data(
                symbol=self.symbol,
                max_candles=self.max_candles,
                progress_callback=progress_callback
            )

            if result:
                self.result.emit({
                    'success': True,
                    'symbol': self.symbol,
                    'message': f"Downloaded {self.max_candles} candles for {self.symbol}"
                })
            else:
                self.result.emit({
                    'success': False,
                    'symbol': self.symbol,
                    'message': f"Failed to download data for {self.symbol}"
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

            # Import SIMPLE model (proven to work)
            from src.ml.models.simple_model import SimpleTrainer, SimpleModelConfig
            from src.core.config import AppConfig
            import numpy as np
            from pathlib import Path

            # Config
            config = SimpleModelConfig(
                hidden_size=int(self.config.get('hidden_size', 32)),
                dropout=float(self.config.get('dropout', 0.1)),
                learning_rate=float(self.config.get('learning_rate', 0.001)),
                batch_size=int(self.config.get('batch_size', 256)),
                max_epochs=int(self.config.get('max_epochs', 50)),
                early_stopping_patience=10,
                sequence_length=12,  # Short sequences work fine
                prediction_horizon=4  # 4h ahead prediction
            )

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

                    print(f"[TRAIN] {symbol}: Loaded {len(df)} rows from {largest_file.name}")

                    # Split chronologically (70/15/15)
                    split1 = int(len(df) * 0.7)
                    split2 = int(len(df) * 0.85)

                    train_df = df.iloc[:split1]
                    val_df = df.iloc[split1:split2]
                    test_df = df.iloc[split2:]

                    # Create trainer and train
                    trainer = SimpleTrainer(config=config)
                    train_results = trainer.train(train_df, val_df, verbose=True)

                    # Evaluate on test
                    self.progress.emit(base_progress + 5, 100, f"Evaluating {symbol}...")
                    test_results = trainer.evaluate(test_df, verbose=True)

                    # Save model
                    model_path = f"models/checkpoints/{symbol}/simple_model.pt"
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    trainer.save(model_path)

                    all_results[symbol] = {
                        'model_path': model_path,
                        'accuracy': test_results['accuracy'],
                        'ic': test_results['ic'],
                        'sharpe_ratio': test_results['sharpe_approx'],
                        'edge': test_results['edge'],
                        'val_acc': train_results['best_val_acc'],
                        'val_ic': train_results['best_val_ic']
                    }

                    # Emit results for this symbol
                    self.epoch_complete.emit(
                        sym_idx + 1,
                        test_results['sharpe_approx'],
                        test_results['ic'],
                        {
                            'symbol': symbol,
                            'accuracy': test_results['accuracy'],
                            'ic': test_results['ic'],
                            'sharpe': test_results['sharpe_approx'],
                            'edge_pct': test_results['edge'] * 100
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
                'message': f"Trained {len(successful)}/{len(self.symbols)} models. Avg Acc: {avg_acc:.1%}, Sharpe: {avg_sharpe:.2f}, IC: {avg_ic:.4f}"
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
                # Walk-forward backtesting using SimpleTrainer (most realistic)
                from src.ml.models.simple_model import SimpleTrainer, SimpleModelConfig
                from src.ml.models.simple_model import create_features, create_targets, get_feature_columns
                from pathlib import Path
                import numpy as np
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

                self.progress.emit(30, 100, "Preparing features...")

                # Create features and targets
                df = create_features(df)
                df = create_targets(df, horizon=4)
                df = df.dropna()

                feature_columns = get_feature_columns(df)

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
                all_predictions = []
                all_actuals = []
                equity_values = [backtest_config.initial_capital]
                current_capital = backtest_config.initial_capital

                config = SimpleModelConfig(
                    hidden_size=32,
                    dropout=0.1,
                    learning_rate=0.001,
                    batch_size=256,
                    max_epochs=30,  # Fewer epochs for walk-forward
                    early_stopping_patience=5,
                    sequence_length=12,
                    prediction_horizon=4
                )

                for fold_idx in range(n_folds):
                    if self._stopped:
                        return

                    fold_start = fold_idx * fold_size
                    train_end = fold_start + train_size
                    test_start = train_end + gap_periods
                    test_end = min(test_start + test_size, len(df))

                    if test_end > len(df):
                        break

                    train_df = df.iloc[fold_start:train_end].copy()
                    test_df = df.iloc[test_start:test_end].copy()

                    self.progress.emit(
                        40 + int((fold_idx / n_folds) * 50),
                        100,
                        f"Fold {fold_idx + 1}/{n_folds}: Training..."
                    )

                    # Train or load model
                    trainer = SimpleTrainer(config=config)

                    if retrain or fold_idx == 0:
                        # Split train into train/val (80/20)
                        val_split = int(len(train_df) * 0.8)
                        fold_train = train_df.iloc[:val_split]
                        fold_val = train_df.iloc[val_split:]

                        train_results = trainer.train(fold_train, fold_val, verbose=False)
                    elif model_path and Path(model_path).exists():
                        trainer = SimpleTrainer.load(model_path)
                    else:
                        # Use previous fold's model
                        pass

                    # Evaluate on test fold
                    test_results = trainer.evaluate(test_df, verbose=False)

                    # Track metrics
                    fold_metrics = {
                        'fold': fold_idx + 1,
                        'accuracy': test_results['accuracy'],
                        'ic': test_results['ic'],
                        'sharpe': test_results['sharpe_approx'],
                        'edge': test_results['edge'],
                        'n_samples': test_results['n_samples']
                    }
                    fold_results.append(fold_metrics)

                    # Simulate trading on test fold
                    # Simple: bet proportional to confidence, win/lose based on accuracy
                    accuracy = test_results['accuracy']
                    n_trades = test_results['n_samples']

                    # Expected return per trade based on accuracy
                    win_rate = accuracy
                    avg_win = 0.002  # 0.2% average win
                    avg_loss = 0.002  # 0.2% average loss

                    for _ in range(min(n_trades, 100)):  # Cap at 100 trades per fold
                        if np.random.random() < win_rate:
                            current_capital *= (1 + avg_win - backtest_config.trade_fee)
                        else:
                            current_capital *= (1 - avg_loss - backtest_config.trade_fee)
                        equity_values.append(current_capital)

                    print(f"  Fold {fold_idx + 1}: Acc={accuracy:.1%}, IC={test_results['ic']:.4f}, "
                          f"Sharpe={test_results['sharpe_approx']:.2f}")

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
                # Model-based backtest (LSTM)
                model_path = self.config.get('model_path')
                if not model_path:
                    raise ValueError("model_path required for model-based backtest")

                self.progress.emit(20, 100, "Loading LSTM model...")

                from src.ml.models.lstm_model import SimpleLSTMTrainer
                import torch
                import numpy as np

                trainer = SimpleLSTMTrainer.load_from_checkpoint(model_path)
                sequence_length = trainer.sequence_length

                self.progress.emit(40, 100, f"Loading data for {symbol}...")

                # Load test data
                from src.ml.preprocessing.preprocessor import CryptoPreprocessor
                preprocessor = CryptoPreprocessor(AppConfig.DATASET_DIR)
                df = preprocessor.load_symbol_data(symbol)
                df = preprocessor.generate_features(df)
                df = df.dropna()

                # Normalize using saved scalers
                scaler_path = f"models/checkpoints/{symbol}_scalers.pkl"
                try:
                    import pickle
                    with open(scaler_path, 'rb') as f:
                        scaler_data = pickle.load(f)
                    preprocessor.scalers = scaler_data.get('scalers', {})
                    preprocessor.target_scaler = scaler_data.get('target_scaler')
                    df = preprocessor.normalize(df, fit=False)
                except:
                    df = preprocessor.normalize(df, fit=True)

                if self._stopped:
                    return

                self.progress.emit(60, 100, "Running model predictions...")

                # Get feature columns
                feature_cols = trainer.feature_columns or [
                    c for c in df.select_dtypes(include='number').columns
                    if c not in ['datetime', 'symbol', 'target_return', 'target_close', 'actual_direction']
                ]

                features = df[feature_cols].values.astype(np.float32)
                predictions = []
                actuals = []

                trainer.model.eval()
                with torch.no_grad():
                    for i in range(sequence_length, len(features)):
                        if self._stopped:
                            return

                        seq = torch.FloatTensor(features[i-sequence_length:i]).unsqueeze(0)
                        class_logits, reg_out = trainer.model(seq)
                        predictions.append(reg_out.numpy().flatten()[0])

                        if 'target_return' in df.columns:
                            actuals.append(df['target_return'].iloc[i])

                        if i % 100 == 0:
                            progress = 60 + int(((i - sequence_length) / (len(features) - sequence_length)) * 30)
                            self.progress.emit(progress, 100, f"Predicting... {i-sequence_length}/{len(features)-sequence_length}")

                predictions = np.array(predictions)
                actuals = np.array(actuals) if actuals else predictions * 0

                # Get prices and timestamps
                test_df = df.iloc[sequence_length:]
                if 'close' in test_df.columns:
                    prices = test_df['close'].values[:len(predictions)]
                else:
                    prices = np.ones(len(predictions)) * 100

                if 'datetime' in test_df.columns:
                    timestamps = test_df['datetime'].iloc[:len(predictions)]
                else:
                    timestamps = pd.date_range(start='2024-01-01', periods=len(predictions), freq='h')

                self.progress.emit(90, 100, "Running backtest...")

                # Set target scaler for denormalization
                if preprocessor.target_scaler:
                    backtester.set_target_scaler(preprocessor.target_scaler)

                # Run backtest with predictions
                results = backtester.run_backtest_from_returns(
                    predicted_returns=predictions,
                    actual_returns=actuals[:len(predictions)],
                    current_prices=prices,
                    timestamps=timestamps,
                    denormalize=True
                )

                self.progress.emit(100, 100, "Backtest complete!")
                self.result.emit(results)

        except Exception as e:
            error_msg = f"Backtest failed: {str(e)}"
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

    def run(self):
        try:
            import pandas as pd
            import numpy as np
            from src.api.ccxt_client import get_ccxt_client

            timeframe = self.config.get('timeframe', '1h')
            past_candles = self.config.get('past_candles', 5)
            strategies = self.config.get('strategies', ['rsi_macd'])
            logic_mode = self.config.get('logic_mode', 'or')  # 'and' or 'or'
            include_buy = self.config.get('include_buy', True)
            include_sell = self.config.get('include_sell', True)

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

                            # Filter to BUY/SELL with confidence > 0
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
