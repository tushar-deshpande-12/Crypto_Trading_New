"""
Prediction Panel - Comprehensive strategy analysis and recommendation

Features:
1. Compares gains from multiple indicator strategy combinations
2. Shows backtesting results with probability percentages
3. Analyzes support/resistance and pivot levels
4. Provides final BUY/SELL/HOLD recommendation
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QFrame, QGroupBox, QGridLayout, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QTextEdit, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QFont
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from src.gui_pyqt.styles import COLORS, get_chart_colors


# Timeframes for analysis
TIMEFRAMES = {
    '15m': '15 Minutes',
    '30m': '30 Minutes',
    '1h': '1 Hour',
    '4h': '4 Hours',
    '1d': '1 Day',
}

# All strategies to analyze (combo strategies + Directionality AI)
STRATEGIES_TO_ANALYZE = [
    ('rsi_macd', 'RSI+MACD'),
    ('bb_rsi', 'BB+RSI'),
    ('macd_ma', 'MACD+MA'),
    ('stoch_rsi', 'Stoch+RSI'),
    ('triple_ema', 'Triple EMA'),
    ('adx_macd', 'ADX+MACD'),
    ('stochastic', 'Stochastic'),
    ('ensemble', 'Ensemble'),
    ('directionality', 'Directionality AI'),  # 9th strategy - LSTM direction predictor
]


@dataclass
class StrategyResult:
    """Result from a single strategy analysis."""
    name: str
    display_name: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    win_rate: float
    avg_return: float
    total_trades: int
    profitable_trades: int
    recent_signal_count: int


@dataclass
class PredictionResult:
    """Overall prediction result."""
    symbol: str
    timeframe: str
    recommendation: str  # BUY, SELL, HOLD
    confidence: float
    strategy_results: List[StrategyResult]
    support_levels: List[float]
    resistance_levels: List[float]
    pivot_levels: Dict[str, float]
    current_price: float
    nearest_support: float
    nearest_resistance: float
    risk_reward_ratio: float
    bull_strategies: int
    bear_strategies: int
    neutral_strategies: int
    analysis_time: str  # When analysis was performed
    data_timestamp: str  # Timestamp of the latest candle data
    # RankNet AI fields
    ranknet_direction: str = '--'  # UP/DOWN/NEUTRAL
    ranknet_confidence: float = 0.0
    ranknet_metrics: Optional[Dict] = None
    ranknet_model_missing: bool = False  # True if no trained model for timeframe


class PredictionWorker(QThread):
    """Worker thread for running prediction analysis."""
    progress = pyqtSignal(int, int, str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, symbol: str, timeframe: str, candle_limit: int):
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.candle_limit = candle_limit
        self._stopped = False

    def run(self):
        try:
            from src.api.ccxt_client import get_ccxt_client
            from src.ml.strategies import get_strategy
            from src.gui_pyqt.utils.chart_calculations import (
                calculate_pivot_points_from_df, PivotType,
                find_support_resistance_levels
            )

            self.progress.emit(0, 100, f"Fetching data for {self.symbol}...")

            # Fetch data
            client = get_ccxt_client('binance')
            ccxt_symbol = client.convert_symbol_format(self.symbol, to_ccxt=True)

            try:
                df = client.get_max_ohlcv(ccxt_symbol, self.timeframe, self.candle_limit)
            except ValueError as e:
                # Symbol doesn't exist or API error
                self.error.emit(f"Symbol '{self.symbol}' not found on Binance. Check the symbol name.")
                return
            except Exception as e:
                self.error.emit(f"Failed to fetch data: {str(e)}")
                return

            if df is None or len(df) < 100:
                # Use centralized delisted symbols list
                from src.gui_pyqt.utils.symbols import DELISTED_SYMBOLS

                # Common symbol rebrandings
                rebrandings = {
                    'MATICUSDT': 'POLUSDT (MATIC rebranded to POL)',
                    'MATIC': 'POL (MATIC rebranded to POL)',
                    'LUNAUSDT': 'LUNC or LUNA2 (LUNA split after crash)',
                    'LUNA': 'LUNC or LUNA2 (LUNA split after crash)',
                    'AGIXUSDT': 'Use FET (merged into ASI alliance)',
                    'OCEANUSDT': 'Use FET (merged into ASI alliance)',
                    'RNDRUSDT': 'Use RENDERUSDT (rebranded)',
                }

                sym_upper = self.symbol.upper()
                suggestion = rebrandings.get(sym_upper, None)

                if suggestion:
                    self.error.emit(f"No data for {self.symbol}. Try {suggestion}")
                elif sym_upper in DELISTED_SYMBOLS or (sym_upper + 'USDT') in DELISTED_SYMBOLS:
                    self.error.emit(f"{self.symbol} has been delisted from Binance. Please choose another symbol.")
                elif len(df) == 0 if df is not None else True:
                    self.error.emit(f"No data for {self.symbol}. Symbol may be delisted or suspended.")
                else:
                    self.error.emit(f"Insufficient data for {self.symbol} ({len(df)} candles, need 100+)")
                return

            # Capture data timestamp (last candle time)
            if isinstance(df.index, pd.DatetimeIndex):
                data_timestamp = df.index[-1].strftime('%Y-%m-%d %H:%M:%S UTC')
            else:
                data_timestamp = "Unknown"

            # Capture analysis time
            analysis_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

            self.progress.emit(10, 100, "Calculating indicators...")

            # Calculate all indicators
            df = self._calculate_all_indicators(df)

            current_price = df['close'].iloc[-1]

            self.progress.emit(20, 100, "Analyzing strategies...")

            # Analyze each strategy
            strategy_results = []
            total_strategies = len(STRATEGIES_TO_ANALYZE)

            for idx, (strategy_key, display_name) in enumerate(STRATEGIES_TO_ANALYZE):
                if self._stopped:
                    return

                progress_pct = 20 + int((idx / total_strategies) * 50)
                self.progress.emit(progress_pct, 100, f"Analyzing {display_name}...")

                try:
                    result = self._analyze_strategy(df, strategy_key, display_name)
                    strategy_results.append(result)
                except Exception as e:
                    print(f"Error analyzing {strategy_key}: {e}")
                    # Add neutral result on error
                    strategy_results.append(StrategyResult(
                        name=strategy_key,
                        display_name=display_name,
                        signal='HOLD',
                        confidence=0,
                        win_rate=0,
                        avg_return=0,
                        total_trades=0,
                        profitable_trades=0,
                        recent_signal_count=0
                    ))

            self.progress.emit(75, 100, "Calculating support/resistance...")

            # Calculate support/resistance levels
            sr_levels = find_support_resistance_levels(df, lookback=10, min_touches=2)
            support_levels = [l.price for l in sr_levels if l.level_type == 'support'][:5]
            resistance_levels = [l.price for l in sr_levels if l.level_type == 'resistance'][:5]

            # Calculate pivot points
            self.progress.emit(80, 100, "Calculating pivot points...")
            pivots = calculate_pivot_points_from_df(df, PivotType.CLASSIC)
            pivot_dict = pivots.to_dict()

            # Find nearest support/resistance
            all_supports = support_levels + [pivot_dict.get('S1', 0), pivot_dict.get('S2', 0)]
            all_resistances = resistance_levels + [pivot_dict.get('R1', 0), pivot_dict.get('R2', 0)]

            all_supports = [s for s in all_supports if s > 0 and s < current_price]
            all_resistances = [r for r in all_resistances if r > 0 and r > current_price]

            nearest_support = max(all_supports) if all_supports else current_price * 0.95
            nearest_resistance = min(all_resistances) if all_resistances else current_price * 1.05

            self.progress.emit(90, 100, "Generating recommendation...")

            # Count strategy signals
            bull_count = sum(1 for r in strategy_results if r.signal == 'BUY' and r.confidence > 0.3)
            bear_count = sum(1 for r in strategy_results if r.signal == 'SELL' and r.confidence > 0.3)
            neutral_count = len(strategy_results) - bull_count - bear_count

            # Calculate weighted recommendation
            total_weight = 0
            weighted_score = 0  # Positive = bullish, negative = bearish

            for result in strategy_results:
                weight = result.confidence * (1 + result.win_rate)
                if result.signal == 'BUY':
                    weighted_score += weight
                elif result.signal == 'SELL':
                    weighted_score -= weight
                total_weight += weight

            # Normalize score to -1 to 1
            if total_weight > 0:
                normalized_score = weighted_score / total_weight
            else:
                normalized_score = 0

            # Determine recommendation
            if normalized_score > 0.2:
                recommendation = 'BUY'
                confidence = min(normalized_score * 0.8 + 0.2, 1.0)
            elif normalized_score < -0.2:
                recommendation = 'SELL'
                confidence = min(abs(normalized_score) * 0.8 + 0.2, 1.0)
            else:
                recommendation = 'HOLD'
                confidence = 1.0 - abs(normalized_score) * 2

            # Calculate risk/reward ratio
            risk = current_price - nearest_support
            reward = nearest_resistance - current_price
            rr_ratio = reward / risk if risk > 0 else 0

            # RankNet AI Direction Analysis - uses pre-trained model if available
            ranknet_direction = '--'
            ranknet_confidence = 0.0
            ranknet_metrics = None
            ranknet_model_missing = False

            try:
                self.progress.emit(92, 100, "Running RankNet AI analysis...")
                from pathlib import Path
                from src.ml.models.ranknet_model import RankNetPredictor

                # Check for pre-trained model (timeframe-specific or legacy)
                model_path = Path(f"models/checkpoints/{self.symbol}/ranknet_{self.timeframe}.pt")
                legacy_path = Path(f"models/checkpoints/{self.symbol}/ranknet_model.pt")

                predictor = None
                if model_path.exists():
                    predictor = RankNetPredictor.load(str(model_path))
                    print(f"[RankNet] Loaded model for {self.symbol} @ {self.timeframe}")
                elif self.timeframe == '1h' and legacy_path.exists():
                    predictor = RankNetPredictor.load(str(legacy_path))
                    print(f"[RankNet] Loaded legacy 1h model for {self.symbol}")

                if predictor is None:
                    # No pre-trained model available
                    ranknet_model_missing = True
                    print(f"[RankNet] No trained model for {self.symbol} @ {self.timeframe}. "
                          "Train in Direction Prediction tab first.")
                else:
                    # Prepare OHLCV dataframe
                    ranknet_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

                    if len(ranknet_df) >= 300:
                        result_rn = predictor.predict(ranknet_df)
                        ranknet_direction = result_rn['direction']
                        ranknet_confidence = result_rn['confidence']
                        ranknet_metrics = result_rn['metrics']
                        print(f"[RankNet] {self.symbol} @ {self.timeframe}: {ranknet_direction} "
                              f"(conf={ranknet_confidence:.2f}, "
                              f"IC={ranknet_metrics['ic']:.4f})")
                    else:
                        print(f"[RankNet] Skipped: insufficient data ({len(ranknet_df)} rows)")
            except Exception as e:
                print(f"[RankNet] Analysis failed: {e}")
                import traceback
                traceback.print_exc()

            # Create final result
            prediction = PredictionResult(
                symbol=self.symbol,
                timeframe=self.timeframe,
                recommendation=recommendation,
                confidence=confidence,
                strategy_results=strategy_results,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                pivot_levels=pivot_dict,
                current_price=current_price,
                nearest_support=nearest_support,
                nearest_resistance=nearest_resistance,
                risk_reward_ratio=rr_ratio,
                bull_strategies=bull_count,
                bear_strategies=bear_count,
                neutral_strategies=neutral_count,
                analysis_time=analysis_time,
                data_timestamp=data_timestamp,
                ranknet_direction=ranknet_direction,
                ranknet_confidence=ranknet_confidence,
                ranknet_metrics=ranknet_metrics,
                ranknet_model_missing=ranknet_model_missing,
            )

            self.progress.emit(100, 100, "Analysis complete!")
            self.result.emit(prediction)

        except Exception as e:
            self.error.emit(f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.finished.emit()

    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators needed for strategies."""
        close = df['close']
        high = df['high']
        low = df['low']

        # Moving Averages
        df['sma_10'] = close.rolling(window=10).mean()
        df['sma_20'] = close.rolling(window=20).mean()
        df['sma_50'] = close.rolling(window=50).mean()
        df['ema_12'] = close.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close.ewm(span=26, adjust=False).mean()

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

        # ADX
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

        # MA ratios
        df['sma10_sma20_ratio'] = (df['sma_10'] / df['sma_20'].replace(0, np.nan)) - 1.0
        df['price_sma10_ratio'] = (close / df['sma_10'].replace(0, np.nan)) - 1.0
        df['price_sma20_ratio'] = (close / df['sma_20'].replace(0, np.nan)) - 1.0

        return df.dropna()

    def _analyze_strategy(self, df: pd.DataFrame, strategy_key: str,
                         display_name: str) -> StrategyResult:
        """Analyze a single strategy and return results."""
        # Handle special strategies separately
        if strategy_key == 'directionality':
            return self._analyze_directionality(df, display_name)

        from src.ml.strategies import get_strategy

        strategy = get_strategy(strategy_key)

        # Prepare data with datetime column
        strategy_data = df.copy()
        if isinstance(strategy_data.index, pd.DatetimeIndex):
            strategy_data['datetime'] = strategy_data.index

        # Generate signals
        signals = strategy.generate_signals(strategy_data)

        # Get recent signals (last 20 candles)
        recent_signals = signals[-20:] if len(signals) >= 20 else signals
        recent_buy = sum(1 for s in recent_signals if s.action == 'BUY' and s.confidence > 0.3)
        recent_sell = sum(1 for s in recent_signals if s.action == 'SELL' and s.confidence > 0.3)

        # Get current signal - find most recent BUY/SELL within last 5 candles
        # This matches the signal scanner logic for consistency
        lookback_signals = signals[-5:] if len(signals) >= 5 else signals
        current_signal = None
        current_action = 'HOLD'
        current_confidence = 0

        # Search backwards for most recent actionable signal
        for sig in reversed(lookback_signals):
            if sig.action in ('BUY', 'SELL') and sig.confidence > 0:
                current_signal = sig
                current_action = sig.action
                current_confidence = sig.confidence
                break

        # If no actionable signal found in lookback, use the last signal
        if current_signal is None and signals:
            current_signal = signals[-1]
            current_action = current_signal.action
            current_confidence = current_signal.confidence

        # Backtest to get win rate
        win_rate, avg_return, total_trades, profitable_trades = self._quick_backtest(
            df, signals
        )

        return StrategyResult(
            name=strategy_key,
            display_name=display_name,
            signal=current_action,
            confidence=current_confidence,
            win_rate=win_rate,
            avg_return=avg_return,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            recent_signal_count=recent_buy - recent_sell  # Positive = bullish bias
        )

    def _quick_backtest(self, df: pd.DataFrame, signals: List) -> tuple:
        """Quick backtest to calculate win rate and returns."""
        if not signals or len(df) < 2:
            return 0.0, 0.0, 0, 0

        trades = []
        in_position = False
        entry_price = 0
        entry_idx = 0

        for i, signal in enumerate(signals):
            if i >= len(df):
                break

            price = df['close'].iloc[i]

            if signal.action == 'BUY' and not in_position and signal.confidence > 0.3:
                in_position = True
                entry_price = price
                entry_idx = i

            elif signal.action == 'SELL' and in_position:
                # Close position
                exit_price = price
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trades.append(pnl_pct)
                in_position = False

        if not trades:
            return 0.0, 0.0, 0, 0

        profitable = sum(1 for t in trades if t > 0)
        win_rate = profitable / len(trades) if trades else 0
        avg_return = np.mean(trades) if trades else 0

        return win_rate, avg_return, len(trades), profitable

    def _analyze_directionality(self, df: pd.DataFrame, display_name: str) -> StrategyResult:
        """
        On-the-fly Directionality AI analysis.
        Trains a SimplePredictor LSTM on the fetched data, then evaluates using
        ver11_gpt-style backtesting (fixed TP/SL with rolling z-score confidence).
        """
        try:
            import torch
            from src.ml.models.simple_model import (
                SimpleModelConfig, SimplePredictor, SimpleTrainer,
                create_features, create_targets, get_feature_columns
            )
            from sklearn.preprocessing import StandardScaler

            # Prepare OHLCV dataframe
            ohlcv_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            if len(ohlcv_df) < 300:
                raise ValueError(f"Insufficient data: {len(ohlcv_df)} rows (need 300+)")

            # Feature engineering (SimpleModel features)
            feat_df = create_features(ohlcv_df)
            feat_df = create_targets(feat_df, horizon=4)
            feat_df = feat_df.dropna().reset_index(drop=True)

            feature_cols = get_feature_columns(feat_df)
            if not feature_cols:
                raise ValueError("No valid feature columns found")

            features = feat_df[feature_cols].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = np.clip(features, -10, 10)

            # Train/val split (70/30 chronological)
            split_idx = int(len(features) * 0.7)
            train_features = features[:split_idx]
            train_targets = feat_df['target'].values[:split_idx]
            val_features = features[split_idx:]
            val_targets = feat_df['target'].values[split_idx:]

            # Normalize
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
            val_features = scaler.transform(val_features)

            # Train SimplePredictor (ver11_gpt hyperparameters)
            config = SimpleModelConfig(
                hidden_size=128, dropout=0.1, learning_rate=0.001,
                batch_size=256, max_epochs=20, early_stopping_patience=10,
                sequence_length=12, prediction_horizon=4,
            )

            from torch.utils.data import DataLoader
            from src.ml.models.simple_model import SimpleDataset

            train_ds = SimpleDataset(train_features, train_targets, seq_len=config.sequence_length)
            val_ds = SimpleDataset(val_features, val_targets, seq_len=config.sequence_length)

            train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SimplePredictor(
                input_size=len(feature_cols),
                hidden_size=config.hidden_size,
                dropout=config.dropout,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
            criterion = torch.nn.BCEWithLogitsLoss()

            # Quick training loop
            best_val_ic = -1.0
            best_state = None
            patience = 0

            for epoch in range(config.max_epochs):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # Validation IC
                model.eval()
                val_preds_list, val_actuals_list = [], []
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device)
                        pred = torch.sigmoid(model(x))
                        val_preds_list.extend(pred.cpu().numpy().flatten())
                        val_actuals_list.extend(y.numpy().flatten())

                val_preds_arr = np.array(val_preds_list)
                val_actuals_arr = np.array(val_actuals_list)
                val_ic = np.corrcoef(val_preds_arr, val_actuals_arr)[0, 1] if len(val_preds_arr) > 2 else 0
                if np.isnan(val_ic):
                    val_ic = 0

                if val_ic > best_val_ic:
                    best_val_ic = val_ic
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= config.early_stopping_patience:
                        break

            if best_state:
                model.load_state_dict(best_state)

            # Generate predictions on FULL dataset for ver11_gpt-style backtest
            all_features = scaler.transform(features)
            model.eval()
            preds_list = []
            with torch.no_grad():
                for i in range(config.sequence_length, len(all_features)):
                    seq = torch.FloatTensor(all_features[i - config.sequence_length:i]).unsqueeze(0).to(device)
                    logit = model(seq)
                    prob = torch.sigmoid(logit).cpu().numpy().flatten()[0]
                    # Convert probability to signed prediction (like RankNet)
                    preds_list.append(prob - 0.5)

            preds_arr = np.array(preds_list)

            # Align dataframe with predictions
            aligned_df = feat_df.iloc[config.sequence_length:].copy().reset_index(drop=True)
            if len(aligned_df) > len(preds_arr):
                aligned_df = aligned_df.iloc[:len(preds_arr)]

            # ver11_gpt-style backtest (fixed TP/SL with confidence filtering)
            from src.ml.models.ranknet_model import backtest_fixed_profit_with_confidence
            bt = backtest_fixed_profit_with_confidence(
                aligned_df, preds_arr,
                confidence_threshold=0.9,
                take_profit=0.03,
                stop_loss=0.0075,
            )

            # Directional accuracy
            future_returns = aligned_df['future_return'].values[:len(preds_arr)] if 'future_return' in aligned_df.columns else np.zeros(len(preds_arr))
            future_returns = np.nan_to_num(future_returns, nan=0.0)
            pred_dir = np.sign(preds_arr)
            true_dir = np.sign(future_returns)
            valid = (pred_dir != 0) & (true_dir != 0)
            dir_acc = float((pred_dir[valid] == true_dir[valid]).mean()) if valid.sum() > 0 else 0.5

            # Determine signal from recent predictions
            recent = preds_arr[-20:] if len(preds_arr) >= 20 else preds_arr
            avg_pred = np.mean(recent)
            if avg_pred > 0.02:
                signal = 'BUY'
                confidence = min(abs(avg_pred) * 10, 0.95)
            elif avg_pred < -0.02:
                signal = 'SELL'
                confidence = min(abs(avg_pred) * 10, 0.95)
            else:
                signal = 'HOLD'
                confidence = 0.3

            win_rate = bt['Win Rate']
            total_trades = bt['Trades']
            profitable_trades = int(win_rate * total_trades)

            # Avg return from equity generated
            avg_return = (bt['Equity Generated'] / bt['Initial Capital'] * 100) if total_trades > 0 else 0.0

            print(f"[Directionality] {signal} (conf={confidence:.2f}, "
                  f"DirAcc={dir_acc:.1%}, WR={win_rate:.1%}, trades={total_trades})")

            return StrategyResult(
                name='directionality',
                display_name=display_name,
                signal=signal,
                confidence=confidence,
                win_rate=win_rate,
                avg_return=avg_return,
                total_trades=total_trades,
                profitable_trades=profitable_trades,
                recent_signal_count=1 if signal == 'BUY' else (-1 if signal == 'SELL' else 0)
            )

        except Exception as e:
            print(f"Directionality analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return StrategyResult(
                name='directionality',
                display_name=display_name,
                signal='HOLD',
                confidence=0,
                win_rate=0,
                avg_return=0,
                total_trades=0,
                profitable_trades=0,
                recent_signal_count=0
            )

    def stop(self):
        self._stopped = True


class PredictionPanel(QWidget):
    """
    Prediction Panel with comprehensive strategy analysis.

    Analyzes multiple strategies, calculates probabilities,
    and provides BUY/SELL/HOLD recommendations.
    """

    train_all_requested = pyqtSignal(str)  # timeframe

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._train_all_worker = None
        self._result: Optional[PredictionResult] = None
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QLabel("AI Strategy Prediction")
        header.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {COLORS['text_primary']};")
        layout.addWidget(header)

        subtitle = QLabel("Compare strategies, analyze levels, and get recommendations")
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(subtitle)

        # Input controls
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)

        # Symbol input
        input_layout.addWidget(QLabel("Symbol:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.setEditable(True)
        self.symbol_combo.setMinimumWidth(120)
        # Load all USDT symbols from Binance (with popular ones first)
        try:
            from src.gui_pyqt.utils.symbols import get_all_usdt_symbols
            symbols = get_all_usdt_symbols()
            self.symbol_combo.addItems(symbols)
        except Exception as e:
            # Fallback to defaults
            from src.gui_pyqt.utils.symbols import get_default_symbols
            self.symbol_combo.addItems(get_default_symbols())
        input_layout.addWidget(self.symbol_combo)

        # Timeframe
        input_layout.addWidget(QLabel("Timeframe:"))
        self.timeframe_combo = QComboBox()
        for key, label in TIMEFRAMES.items():
            self.timeframe_combo.addItem(label, key)
        self.timeframe_combo.setCurrentText('1 Hour')
        input_layout.addWidget(self.timeframe_combo)

        # Candle limit
        input_layout.addWidget(QLabel("Candles:"))
        self.candle_spin = QSpinBox()
        self.candle_spin.setRange(200, 2000)
        self.candle_spin.setValue(500)
        self.candle_spin.setSingleStep(100)
        input_layout.addWidget(self.candle_spin)

        input_layout.addStretch()

        # Analyze button
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setMinimumWidth(120)
        self.analyze_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent']};
            }}
        """)
        self.analyze_btn.clicked.connect(self._on_analyze_clicked)
        input_layout.addWidget(self.analyze_btn)

        layout.addWidget(input_frame)

        # Train All section
        train_all_frame = QFrame()
        train_all_frame.setStyleSheet(f"""
            QFrame {{
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 8px;
            }}
        """)
        train_all_layout = QHBoxLayout(train_all_frame)
        train_all_layout.setContentsMargins(12, 8, 12, 8)

        train_all_label = QLabel("Batch Training:")
        train_all_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-weight: bold; border: none;")
        train_all_layout.addWidget(train_all_label)

        # Timeframe selector for train all
        train_all_layout.addWidget(QLabel("Timeframe:"))
        self.train_all_tf_combo = QComboBox()
        for key, label in TIMEFRAMES.items():
            self.train_all_tf_combo.addItem(label, key)
        self.train_all_tf_combo.setCurrentText('1 Hour')
        self.train_all_tf_combo.setMinimumWidth(100)
        train_all_layout.addWidget(self.train_all_tf_combo)

        train_all_layout.addStretch()

        # Train All button
        self.train_all_btn = QPushButton("Train All Cryptos")
        self.train_all_btn.setMinimumWidth(200)
        self.train_all_btn.setMinimumHeight(45)
        self.train_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 24px;
                border-radius: 6px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_secondary']};
            }}
        """)
        self.train_all_btn.clicked.connect(self._on_train_all_clicked)
        train_all_layout.addWidget(self.train_all_btn)

        # Stop button for train all
        self.stop_train_all_btn = QPushButton("Stop")
        self.stop_train_all_btn.setMinimumHeight(45)
        self.stop_train_all_btn.setVisible(False)
        self.stop_train_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                font-weight: bold;
                padding: 10px 16px;
                border-radius: 6px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {COLORS['danger_hover']};
            }}
        """)
        self.stop_train_all_btn.clicked.connect(self._on_stop_train_all)
        train_all_layout.addWidget(self.stop_train_all_btn)

        layout.addWidget(train_all_frame)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.status_label)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side - Recommendation and levels
        left_widget = self._create_recommendation_panel()
        splitter.addWidget(left_widget)

        # Right side - Strategy table
        right_widget = self._create_strategy_table()
        splitter.addWidget(right_widget)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter, stretch=1)

    def _create_recommendation_panel(self) -> QWidget:
        """Create the recommendation display panel."""
        widget = QFrame()
        layout = QVBoxLayout(widget)

        # Main recommendation box
        rec_group = QGroupBox("Recommendation")
        rec_layout = QVBoxLayout(rec_group)

        # Big recommendation label
        self.recommendation_label = QLabel("--")
        self.recommendation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recommendation_label.setStyleSheet(f"""
            font-size: 48px;
            font-weight: bold;
            color: {COLORS['text_secondary']};
            padding: 20px;
        """)
        rec_layout.addWidget(self.recommendation_label)

        # Confidence
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet(f"font-size: 16px; color: {COLORS['text_secondary']};")
        rec_layout.addWidget(self.confidence_label)

        # Strategy breakdown
        self.breakdown_label = QLabel("Bullish: -- | Bearish: -- | Neutral: --")
        self.breakdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.breakdown_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        rec_layout.addWidget(self.breakdown_label)

        # Timestamp display
        timestamp_frame = QFrame()
        timestamp_layout = QVBoxLayout(timestamp_frame)
        timestamp_layout.setContentsMargins(0, 10, 0, 0)

        self.data_timestamp_label = QLabel("Data as of: --")
        self.data_timestamp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.data_timestamp_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        timestamp_layout.addWidget(self.data_timestamp_label)

        self.analysis_timestamp_label = QLabel("Analysis performed: --")
        self.analysis_timestamp_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analysis_timestamp_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        timestamp_layout.addWidget(self.analysis_timestamp_label)

        rec_layout.addWidget(timestamp_frame)

        layout.addWidget(rec_group)

        # Price levels group
        levels_group = QGroupBox("Price Levels")
        levels_layout = QGridLayout(levels_group)

        # Current price
        levels_layout.addWidget(QLabel("Current Price:"), 0, 0)
        self.current_price_label = QLabel("--")
        self.current_price_label.setStyleSheet("font-weight: bold;")
        levels_layout.addWidget(self.current_price_label, 0, 1)

        # Nearest resistance
        levels_layout.addWidget(QLabel("Nearest Resistance:"), 1, 0)
        self.resistance_label = QLabel("--")
        self.resistance_label.setStyleSheet(f"color: {COLORS['chart_red']};")
        levels_layout.addWidget(self.resistance_label, 1, 1)

        # Nearest support
        levels_layout.addWidget(QLabel("Nearest Support:"), 2, 0)
        self.support_label = QLabel("--")
        self.support_label.setStyleSheet(f"color: {COLORS['chart_green']};")
        levels_layout.addWidget(self.support_label, 2, 1)

        # Risk/Reward
        levels_layout.addWidget(QLabel("Risk/Reward Ratio:"), 3, 0)
        self.rr_label = QLabel("--")
        levels_layout.addWidget(self.rr_label, 3, 1)

        layout.addWidget(levels_group)

        # Pivot points group
        pivot_group = QGroupBox("Pivot Points (Daily)")
        pivot_layout = QGridLayout(pivot_group)

        self.pivot_labels = {}
        pivot_names = ['R3', 'R2', 'R1', 'P', 'S1', 'S2', 'S3']
        for i, name in enumerate(pivot_names):
            lbl = QLabel(f"{name}:")
            pivot_layout.addWidget(lbl, i, 0)
            val_lbl = QLabel("--")
            if name.startswith('R'):
                val_lbl.setStyleSheet(f"color: {COLORS['chart_red']};")
            elif name.startswith('S'):
                val_lbl.setStyleSheet(f"color: {COLORS['chart_green']};")
            self.pivot_labels[name] = val_lbl
            pivot_layout.addWidget(val_lbl, i, 1)

        layout.addWidget(pivot_group)

        # Analysis summary
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setStyleSheet(f"""
            background-color: {COLORS['bg_dark']};
            color: {COLORS['text_primary']};
            border: none;
        """)
        summary_layout.addWidget(self.summary_text)

        layout.addWidget(summary_group)

        # RankNet AI Analysis group
        ranknet_group = QGroupBox("RankNet AI Analysis")
        ranknet_layout = QGridLayout(ranknet_group)

        # Direction
        ranknet_layout.addWidget(QLabel("Direction:"), 0, 0)
        self.ranknet_direction_label = QLabel("--")
        self.ranknet_direction_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['text_secondary']};")
        ranknet_layout.addWidget(self.ranknet_direction_label, 0, 1)

        # Confidence
        ranknet_layout.addWidget(QLabel("Confidence:"), 0, 2)
        self.ranknet_confidence_label = QLabel("--")
        self.ranknet_confidence_label.setStyleSheet(f"font-weight: bold; color: {COLORS['accent']};")
        ranknet_layout.addWidget(self.ranknet_confidence_label, 0, 3)

        # ICE
        ranknet_layout.addWidget(QLabel("ICE (Spearman):"), 1, 0)
        self.ranknet_ice_label = QLabel("--")
        self.ranknet_ice_label.setStyleSheet(f"color: {COLORS['accent']};")
        ranknet_layout.addWidget(self.ranknet_ice_label, 1, 1)

        # Directional Accuracy
        ranknet_layout.addWidget(QLabel("Dir. Accuracy:"), 1, 2)
        self.ranknet_dir_acc_label = QLabel("--")
        self.ranknet_dir_acc_label.setStyleSheet(f"color: {COLORS['accent']};")
        ranknet_layout.addWidget(self.ranknet_dir_acc_label, 1, 3)

        # Sharpe
        ranknet_layout.addWidget(QLabel("Sharpe Ratio:"), 2, 0)
        self.ranknet_sharpe_label = QLabel("--")
        self.ranknet_sharpe_label.setStyleSheet(f"color: {COLORS['accent']};")
        ranknet_layout.addWidget(self.ranknet_sharpe_label, 2, 1)

        # Win Rate
        ranknet_layout.addWidget(QLabel("Win Rate:"), 2, 2)
        self.ranknet_winrate_label = QLabel("--")
        self.ranknet_winrate_label.setStyleSheet(f"color: {COLORS['accent']};")
        ranknet_layout.addWidget(self.ranknet_winrate_label, 2, 3)

        # Market Coverage
        ranknet_layout.addWidget(QLabel("Market Coverage:"), 3, 0)
        self.ranknet_coverage_label = QLabel("--")
        self.ranknet_coverage_label.setStyleSheet(f"color: {COLORS['accent']};")
        ranknet_layout.addWidget(self.ranknet_coverage_label, 3, 1)

        # Backtest header
        bt_header = QLabel("Backtest Results")
        bt_header.setStyleSheet("font-weight: bold; margin-top: 4px;")
        ranknet_layout.addWidget(bt_header, 4, 0, 1, 4)

        # Trades
        ranknet_layout.addWidget(QLabel("Trades:"), 5, 0)
        self.ranknet_trades_label = QLabel("--")
        ranknet_layout.addWidget(self.ranknet_trades_label, 5, 1)

        # Equity Generated
        ranknet_layout.addWidget(QLabel("Equity Generated:"), 5, 2)
        self.ranknet_equity_label = QLabel("--")
        ranknet_layout.addWidget(self.ranknet_equity_label, 5, 3)

        # Final Capital
        ranknet_layout.addWidget(QLabel("Final Capital:"), 6, 0)
        self.ranknet_capital_label = QLabel("--")
        self.ranknet_capital_label.setStyleSheet("font-weight: bold;")
        ranknet_layout.addWidget(self.ranknet_capital_label, 6, 1)

        layout.addWidget(ranknet_group)

        layout.addStretch()
        return widget

    def _create_strategy_table(self) -> QWidget:
        """Create the strategy comparison table."""
        widget = QFrame()
        layout = QVBoxLayout(widget)

        # Table group
        table_group = QGroupBox("Strategy Comparison")
        table_layout = QVBoxLayout(table_group)

        self.strategy_table = QTableWidget()
        self.strategy_table.setColumnCount(7)
        self.strategy_table.setHorizontalHeaderLabels([
            'Strategy', 'Signal', 'Confidence', 'Win Rate', 'Avg Return', 'Trades', 'Probability'
        ])

        # Style the table
        self.strategy_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
                gridline-color: {COLORS['border']};
            }}
            QTableWidget::item {{
                padding: 5px;
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_primary']};
                padding: 5px;
                border: 1px solid {COLORS['border']};
            }}
        """)

        header = self.strategy_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, 7):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

        table_layout.addWidget(self.strategy_table)

        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Signal Colors:"))

        buy_label = QLabel(" BUY ")
        buy_label.setStyleSheet(f"background-color: {COLORS['chart_green']}; color: white; padding: 2px 8px;")
        legend_layout.addWidget(buy_label)

        sell_label = QLabel(" SELL ")
        sell_label.setStyleSheet(f"background-color: {COLORS['chart_red']}; color: white; padding: 2px 8px;")
        legend_layout.addWidget(sell_label)

        hold_label = QLabel(" HOLD ")
        hold_label.setStyleSheet(f"background-color: {COLORS['text_secondary']}; color: white; padding: 2px 8px;")
        legend_layout.addWidget(hold_label)

        legend_layout.addStretch()
        table_layout.addLayout(legend_layout)

        layout.addWidget(table_group)

        return widget

    def set_symbols(self, symbols: List[str]):
        """Set available symbols in the combo box."""
        current = self.symbol_combo.currentText()
        self.symbol_combo.clear()
        self.symbol_combo.addItems(symbols)  # All available symbols
        if current in symbols:
            self.symbol_combo.setCurrentText(current)

    def _on_analyze_clicked(self):
        """Handle analyze button click."""
        if self._worker and self._worker.isRunning():
            return

        symbol = self.symbol_combo.currentText().strip().upper()
        if not symbol:
            self.status_label.setText("Please enter a symbol")
            return

        timeframe = self.timeframe_combo.currentData()
        candles = self.candle_spin.value()

        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Analyzing {symbol}...")

        # Clear previous results
        self._clear_results()

        # Start worker
        self._worker = PredictionWorker(symbol, timeframe, candles)
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def _on_result(self, result: PredictionResult):
        """Handle analysis result."""
        self._result = result
        self._display_results(result)

    def _on_error(self, error: str):
        """Handle error."""
        self.status_label.setText(f"Error: {error}")

    def _on_finished(self):
        """Handle worker finished."""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def _clear_results(self):
        """Clear all result displays."""
        self._result = None

        self.recommendation_label.setText("--")
        self.recommendation_label.setStyleSheet(f"""
            font-size: 48px; font-weight: bold;
            color: {COLORS['text_secondary']}; padding: 20px;
        """)
        self.confidence_label.setText("Confidence: --")
        self.breakdown_label.setText("Bullish: -- | Bearish: -- | Neutral: --")
        self.data_timestamp_label.setText("Data as of: --")
        self.analysis_timestamp_label.setText("Analysis performed: --")
        self.current_price_label.setText("--")
        self.resistance_label.setText("--")
        self.support_label.setText("--")
        self.rr_label.setText("--")
        for lbl in self.pivot_labels.values():
            lbl.setText("--")
        self.summary_text.clear()
        self.strategy_table.setRowCount(0)

        # Clear RankNet fields
        self.ranknet_direction_label.setText("--")
        self.ranknet_direction_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {COLORS['text_secondary']};")
        self.ranknet_confidence_label.setText("--")
        self.ranknet_ice_label.setText("--")
        self.ranknet_dir_acc_label.setText("--")
        self.ranknet_sharpe_label.setText("--")
        self.ranknet_winrate_label.setText("--")
        self.ranknet_coverage_label.setText("--")
        self.ranknet_trades_label.setText("--")
        self.ranknet_equity_label.setText("--")
        self.ranknet_capital_label.setText("--")

    def _display_results(self, result: PredictionResult):
        """Display the analysis results."""
        # Recommendation
        rec_colors = {
            'BUY': COLORS['chart_green'],
            'SELL': COLORS['chart_red'],
            'HOLD': COLORS['text_secondary']
        }
        color = rec_colors.get(result.recommendation, COLORS['text_secondary'])

        self.recommendation_label.setText(result.recommendation)
        self.recommendation_label.setStyleSheet(f"""
            font-size: 48px; font-weight: bold;
            color: {color}; padding: 20px;
        """)

        self.confidence_label.setText(f"Confidence: {result.confidence*100:.1f}%")
        self.breakdown_label.setText(
            f"Bullish: {result.bull_strategies} | "
            f"Bearish: {result.bear_strategies} | "
            f"Neutral: {result.neutral_strategies}"
        )

        # Timestamps
        self.data_timestamp_label.setText(f"Data as of: {result.data_timestamp}")
        self.analysis_timestamp_label.setText(f"Analysis performed: {result.analysis_time}")

        # Price levels
        self.current_price_label.setText(f"${result.current_price:,.2f}")
        self.resistance_label.setText(f"${result.nearest_resistance:,.2f}")
        self.support_label.setText(f"${result.nearest_support:,.2f}")

        rr_color = COLORS['chart_green'] if result.risk_reward_ratio >= 2 else COLORS['text_secondary']
        self.rr_label.setText(f"{result.risk_reward_ratio:.2f}")
        self.rr_label.setStyleSheet(f"color: {rr_color};")

        # Pivot points
        for name, label in self.pivot_labels.items():
            value = result.pivot_levels.get(name, 0)
            if value > 0:
                label.setText(f"${value:,.2f}")
            else:
                label.setText("--")

        # Strategy table
        self.strategy_table.setRowCount(len(result.strategy_results))

        for row, sr in enumerate(result.strategy_results):
            # Strategy name
            self.strategy_table.setItem(row, 0, QTableWidgetItem(sr.display_name))

            # Signal with color
            signal_item = QTableWidgetItem(sr.signal)
            if sr.signal == 'BUY':
                signal_item.setBackground(QColor(COLORS['chart_green']))
                signal_item.setForeground(QColor('white'))
            elif sr.signal == 'SELL':
                signal_item.setBackground(QColor(COLORS['chart_red']))
                signal_item.setForeground(QColor('white'))
            signal_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.strategy_table.setItem(row, 1, signal_item)

            # Confidence
            conf_item = QTableWidgetItem(f"{sr.confidence*100:.0f}%")
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.strategy_table.setItem(row, 2, conf_item)

            # Win rate
            wr_item = QTableWidgetItem(f"{sr.win_rate*100:.1f}%")
            wr_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if sr.win_rate > 0.55:
                wr_item.setForeground(QColor(COLORS['chart_green']))
            elif sr.win_rate < 0.45:
                wr_item.setForeground(QColor(COLORS['chart_red']))
            self.strategy_table.setItem(row, 3, wr_item)

            # Avg return
            ret_item = QTableWidgetItem(f"{sr.avg_return:+.2f}%")
            ret_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if sr.avg_return > 0:
                ret_item.setForeground(QColor(COLORS['chart_green']))
            elif sr.avg_return < 0:
                ret_item.setForeground(QColor(COLORS['chart_red']))
            self.strategy_table.setItem(row, 4, ret_item)

            # Trades
            trades_item = QTableWidgetItem(str(sr.total_trades))
            trades_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.strategy_table.setItem(row, 5, trades_item)

            # Probability (win rate adjusted by confidence)
            prob = sr.win_rate * sr.confidence * 100
            prob_item = QTableWidgetItem(f"{prob:.0f}%")
            prob_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.strategy_table.setItem(row, 6, prob_item)

        # Summary text
        summary = self._generate_summary(result)
        self.summary_text.setHtml(summary)

        # RankNet AI Analysis section
        if result.ranknet_model_missing:
            # No trained model for this timeframe - show warning
            self.ranknet_direction_label.setText("No Model")
            self.ranknet_direction_label.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {COLORS['chart_red']};"
            )
            self.ranknet_confidence_label.setText(f"Train model for {result.timeframe} first")
            self.ranknet_confidence_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
            self.ranknet_ice_label.setText("--")
            self.ranknet_dir_acc_label.setText("--")
            self.ranknet_sharpe_label.setText("--")
            self.ranknet_winrate_label.setText("--")
            self.ranknet_coverage_label.setText("--")
            self.ranknet_trades_label.setText("--")
            self.ranknet_equity_label.setText("--")
            self.ranknet_capital_label.setText("--")
        elif result.ranknet_direction and result.ranknet_direction != '--':
            dir_colors = {
                'UP': COLORS['chart_green'],
                'DOWN': COLORS['chart_red'],
                'NEUTRAL': COLORS['text_secondary'],
            }
            dir_color = dir_colors.get(result.ranknet_direction, COLORS['text_secondary'])
            dir_arrows = {'UP': '▲ UP', 'DOWN': '▼ DOWN', 'NEUTRAL': '◆ NEUTRAL'}
            self.ranknet_direction_label.setText(dir_arrows.get(result.ranknet_direction, result.ranknet_direction))
            self.ranknet_direction_label.setStyleSheet(
                f"font-size: 18px; font-weight: bold; color: {dir_color};"
            )
            self.ranknet_confidence_label.setText(f"{result.ranknet_confidence * 100:.0f}%")
            self.ranknet_confidence_label.setStyleSheet(f"font-weight: bold; color: {COLORS['accent']};")

            if result.ranknet_metrics:
                m = result.ranknet_metrics
                self.ranknet_ice_label.setText(f"{m.get('ic', 0):.4f}")
                self.ranknet_dir_acc_label.setText(f"{m.get('directional_accuracy', 0) * 100:.1f}%")
                self.ranknet_sharpe_label.setText(f"{m.get('sharpe', 0):.2f}")
                self.ranknet_winrate_label.setText(f"{m.get('win_rate', 0) * 100:.1f}%")
                self.ranknet_coverage_label.setText(f"{m.get('market_coverage', 0) * 100:.1f}%")
                self.ranknet_trades_label.setText(str(m.get('trades', 0)))

                eq_gen = m.get('equity_generated', 0)
                eq_color = COLORS['chart_green'] if eq_gen >= 0 else COLORS['chart_red']
                self.ranknet_equity_label.setText(f"${eq_gen:+,.2f}")
                self.ranknet_equity_label.setStyleSheet(f"color: {eq_color};")

                self.ranknet_capital_label.setText(f"${m.get('final_capital', 10000):,.2f}")

        self.status_label.setText(f"Analysis complete for {result.symbol}")

    def _on_train_all_clicked(self):
        """Handle Train All button click."""
        if self._train_all_worker and self._train_all_worker.isRunning():
            return

        timeframe = self.train_all_tf_combo.currentData()

        self.train_all_btn.setEnabled(False)
        self.stop_train_all_btn.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Starting batch training on {timeframe}...")

        from src.gui_pyqt.workers import TrainAllWorker

        self._train_all_worker = TrainAllWorker(timeframe=timeframe)
        self._train_all_worker.progress.connect(self._on_train_all_progress)
        self._train_all_worker.symbol_complete.connect(self._on_train_all_symbol)
        self._train_all_worker.result.connect(self._on_train_all_result)
        self._train_all_worker.error.connect(self._on_train_all_error)
        self._train_all_worker.finished.connect(self._on_train_all_finished)
        self._train_all_worker.start()

    def _on_stop_train_all(self):
        """Stop the Train All worker."""
        if self._train_all_worker and self._train_all_worker.isRunning():
            self._train_all_worker.stop()
            self.status_label.setText("Stopping batch training...")

    def _on_train_all_progress(self, current: int, total: int, message: str):
        """Handle Train All progress."""
        if total > 0:
            self.progress_bar.setValue(int((current / total) * 100))
        self.status_label.setText(message)

    def _on_train_all_symbol(self, data: dict):
        """Handle per-symbol completion during Train All."""
        symbol = data.get('symbol', '')
        direction = data.get('direction', '--')
        ic = data.get('ic', 0)
        win_rate = data.get('win_rate', 0)
        self.status_label.setText(
            f"Completed {symbol}: {direction} (IC={ic:.4f}, WR={win_rate:.1%})"
        )

    def _on_train_all_result(self, result: dict):
        """Handle Train All completion."""
        msg = result.get('message', 'Done')
        self.status_label.setText(msg)

    def _on_train_all_error(self, error: str):
        """Handle Train All error."""
        self.status_label.setText(f"Error: {error}")

    def _on_train_all_finished(self):
        """Handle Train All worker finished."""
        self.train_all_btn.setEnabled(True)
        self.stop_train_all_btn.setVisible(False)
        self.progress_bar.setVisible(False)

    def _generate_summary(self, result: PredictionResult) -> str:
        """Generate analysis summary text."""
        lines = []

        # Overall sentiment
        if result.recommendation == 'BUY':
            lines.append(f"<b style='color:{COLORS['chart_green']}'>BULLISH OUTLOOK</b>")
            lines.append(f"{result.bull_strategies} out of {len(result.strategy_results)} strategies signal BUY.")
        elif result.recommendation == 'SELL':
            lines.append(f"<b style='color:{COLORS['chart_red']}'>BEARISH OUTLOOK</b>")
            lines.append(f"{result.bear_strategies} out of {len(result.strategy_results)} strategies signal SELL.")
        else:
            lines.append(f"<b>NEUTRAL - CONSOLIDATION</b>")
            lines.append("Mixed signals suggest waiting for clearer direction.")

        lines.append("")

        # Support/Resistance analysis
        price = result.current_price
        support_dist = (price - result.nearest_support) / price * 100
        resist_dist = (result.nearest_resistance - price) / price * 100

        lines.append("<b>Level Analysis:</b>")
        if support_dist < 2:
            lines.append(f"<span style='color:{COLORS['chart_green']}'>Price near strong support (${result.nearest_support:,.2f})</span>")
        if resist_dist < 2:
            lines.append(f"<span style='color:{COLORS['chart_red']}'>Price near resistance (${result.nearest_resistance:,.2f})</span>")

        # Risk/Reward
        if result.risk_reward_ratio >= 2:
            lines.append(f"<span style='color:{COLORS['chart_green']}'>Favorable R/R ratio: {result.risk_reward_ratio:.1f}:1</span>")
        elif result.risk_reward_ratio < 1:
            lines.append(f"<span style='color:{COLORS['chart_red']}'>Poor R/R ratio: {result.risk_reward_ratio:.1f}:1</span>")

        # Best performing strategies
        sorted_strats = sorted(result.strategy_results,
                              key=lambda x: x.win_rate * x.confidence, reverse=True)
        top_strats = [s for s in sorted_strats[:3] if s.win_rate > 0.5]

        if top_strats:
            lines.append("")
            lines.append("<b>Top Strategies:</b>")
            for s in top_strats:
                lines.append(f"- {s.display_name}: {s.signal} ({s.win_rate*100:.0f}% win rate)")

        return "<br>".join(lines)
