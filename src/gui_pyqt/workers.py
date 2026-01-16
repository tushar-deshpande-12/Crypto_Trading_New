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

            # Import ML modules
            from src.ml.preprocessing.preprocessor import CryptoPreprocessor
            from src.ml.training.trainer import TFTTrainer
            from src.ml.models.model_config import TFTConfig
            from src.core.config import AppConfig

            # Preprocessing
            self.progress.emit(5, 100, "Loading and preprocessing data...")

            preprocessor = CryptoPreprocessor(AppConfig.DATASET_DIR)
            train_df, val_df, test_df = preprocessor.process_all(
                self.symbols,
                save_scaler_path="models/checkpoints/scalers.pkl"
            )

            if self._stopped:
                return

            self.progress.emit(15, 100, "Creating datasets...")

            # Create model config
            max_epochs = int(self.config.get('max_epochs', 50))
            tft_config = TFTConfig(
                hidden_size=int(self.config.get('hidden_size', 32)),
                lstm_layers=int(self.config.get('lstm_layers', 1)),
                attention_heads=int(self.config.get('attention_head_size', 4)),
                dropout=float(self.config.get('dropout', 0.4)),
                learning_rate=float(self.config.get('learning_rate', 0.0005)),
                batch_size=int(self.config.get('batch_size', 128)),
                max_epochs=max_epochs
            )

            self.progress.emit(20, 100, "Setting up trainer...")

            # Use GPU if available and requested
            use_gpu = self.config.get('use_gpu', False)

            # Initialize trainer
            trainer = TFTTrainer(
                config=tft_config,
                checkpoint_dir="models/checkpoints",
                log_dir="logs/training"
            )

            # Training callback
            def epoch_callback(epoch, train_loss, val_loss, metrics=None):
                if self._stopped:
                    raise StopIteration("Training stopped by user")

                progress = 20 + int((epoch / max_epochs) * 75)
                self.progress.emit(progress, 100, f"Epoch {epoch}/{max_epochs}")
                self.epoch_complete.emit(
                    epoch, train_loss, val_loss,
                    metrics or {'train_loss': train_loss, 'val_loss': val_loss}
                )

            # Train
            architecture = self.config.get('architecture', 'lstm')
            trainer.train(
                train_df, val_df,
                architecture=architecture,
                epoch_callback=epoch_callback
            )

            if self._stopped:
                return

            self.progress.emit(95, 100, "Saving model...")

            # Save model
            model_path = trainer.save_model()

            self.progress.emit(100, 100, "Training complete!")
            self.result.emit({
                'success': True,
                'model_path': str(model_path),
                'message': f"Model saved to {model_path}"
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

            else:
                # Model-based backtest
                model_path = self.config.get('model_path')
                if not model_path:
                    raise ValueError("model_path required for model-based backtest")

                self.progress.emit(20, 100, "Loading model...")

                from src.ml.inference.predictor import CryptoPredictor
                predictor = CryptoPredictor(model_path)

                self.progress.emit(40, 100, f"Loading data for {symbol}...")

                # Load test data
                from src.ml.preprocessing.preprocessor import CryptoPreprocessor
                preprocessor = CryptoPreprocessor(AppConfig.DATASET_DIR)
                df = preprocessor.load_symbol_data(symbol)
                df = preprocessor.generate_technical_indicators(df)
                df = df.dropna()

                if self._stopped:
                    return

                self.progress.emit(60, 100, "Running model predictions...")

                # Get predictions for backtest
                predictions = []
                for i in range(len(df) - 168):  # Assuming 168 context length
                    if self._stopped:
                        return
                    pred = predictor.predict_single(df.iloc[i:i+168])
                    predictions.append(pred)

                    if i % 100 == 0:
                        progress = 60 + int((i / (len(df) - 168)) * 30)
                        self.progress.emit(progress, 100, f"Predicting... {i}/{len(df)-168}")

                self.progress.emit(90, 100, "Running backtest...")

                # Run backtest with predictions
                results = backtester.run_backtest(df.iloc[168:], predictions)

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
