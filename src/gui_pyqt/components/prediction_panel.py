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

# All strategies to analyze (combo strategies only)
STRATEGIES_TO_ANALYZE = [
    ('rsi_macd', 'RSI+MACD'),
    ('bb_rsi', 'BB+RSI'),
    ('macd_ma', 'MACD+MA'),
    ('stoch_rsi', 'Stoch+RSI'),
    ('triple_ema', 'Triple EMA'),
    ('adx_macd', 'ADX+MACD'),
    ('stochastic', 'Stochastic'),
    ('ensemble', 'Ensemble'),
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
                self.error.emit(f"Insufficient data for {self.symbol} ({len(df) if df is not None else 0} candles, need 100+)")
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
                data_timestamp=data_timestamp
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

    def stop(self):
        self._stopped = True


class PredictionPanel(QWidget):
    """
    Prediction Panel with comprehensive strategy analysis.

    Analyzes multiple strategies, calculates probabilities,
    and provides BUY/SELL/HOLD recommendations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
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
        self.symbol_combo.addItems(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'])
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

        self.status_label.setText(f"Analysis complete for {result.symbol}")

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
