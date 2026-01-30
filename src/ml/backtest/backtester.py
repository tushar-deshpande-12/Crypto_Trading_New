"""
Backtesting Module for Cryptocurrency Trading Strategy
Tests model predictions on historical data with realistic trading simulation

IMPORTANT: This backtester works with RETURN predictions (target_return)
not raw price predictions. It converts predicted returns to price movements.

Supports both:
1. Model prediction-based backtesting (original)
2. Strategy-based backtesting (new - uses pluggable strategies)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pathlib import Path
from src.ml.training.model_evaluator import safe_direction_accuracy
from src.ml.training.metrics import (
    information_coefficient,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
)

if TYPE_CHECKING:
    from src.ml.strategies.base import BaseStrategy, TradeSignal


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 1000.0  # Starting capital in USD
    trade_fee: float = 0.001  # 0.1% fee per trade
    min_trade_amount: float = 10.0  # Minimum trade size
    confidence_threshold: float = 0.002  # Min predicted change to trade (0.2% - lowered for more trades)


@dataclass
class Trade:
    """Single trade record"""
    timestamp: pd.Timestamp
    action: str  # 'BUY', 'SELL', 'HOLD'
    price: float
    amount: float
    value: float
    balance_after: float
    prediction: float
    actual_next: float


@dataclass
class BacktestResults:
    """Results from backtesting"""
    # Performance metrics
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float

    # Trade statistics
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate: float
    profit_loss_ratio: float

    # Prediction accuracy
    prediction_mae: float
    prediction_rmse: float
    prediction_r2: float
    direction_accuracy: float

    # NEW: Risk-adjusted metrics (ICE and Sharpe focus)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    information_coefficient: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    # Trade history
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Buy and hold comparison
    buy_hold_return: float = 0.0
    buy_hold_return_pct: float = 0.0
    excess_return: float = 0.0


class CryptoBacktester:
    """
    Backtesting engine for cryptocurrency trading strategies

    Simulates trading based on model predictions and calculates:
    - Portfolio value over time
    - Win/loss statistics
    - Prediction accuracy
    - Risk metrics

    IMPORTANT: This backtester supports both:
    1. Price predictions (raw prices in USD)
    2. Return predictions (normalized returns) - requires target_scaler for denormalization
    """

    def __init__(self, config: Optional[BacktestConfig] = None, verbose: bool = True):
        """
        Initialize backtester

        Args:
            config: Backtesting configuration
            verbose: Print progress
        """
        self.config = config or BacktestConfig()
        self.verbose = verbose
        self.target_scaler = None  # For denormalizing return predictions

        if self.verbose:
            print(f"\n[BACKTEST] Initializing backtester")
            print(f"[BACKTEST]   - Initial capital: ${self.config.initial_capital:,.2f}")
            print(f"[BACKTEST]   - Trade fee: {self.config.trade_fee*100:.2f}%")

    def set_target_scaler(self, scaler):
        """
        Set the target scaler for denormalizing return predictions.

        Args:
            scaler: sklearn StandardScaler fitted on target_return
        """
        self.target_scaler = scaler
        if self.verbose:
            print(f"[BACKTEST] Target scaler set for denormalization")

    def denormalize_returns(self, normalized_returns: np.ndarray) -> np.ndarray:
        """
        Denormalize return predictions back to actual returns.

        Args:
            normalized_returns: Normalized return predictions

        Returns:
            Actual returns (as percentages)
        """
        if self.target_scaler is None:
            if self.verbose:
                print(f"[BACKTEST] [!] No target scaler - assuming returns are already denormalized")
            return normalized_returns

        # Denormalize: actual = normalized * scale + mean
        actual_returns = normalized_returns * self.target_scaler.scale_[0] + self.target_scaler.mean_[0]
        return actual_returns

    def returns_to_prices(
        self,
        predicted_returns: np.ndarray,
        current_prices: np.ndarray
    ) -> np.ndarray:
        """
        Convert return predictions to price predictions.

        Args:
            predicted_returns: Predicted returns (as decimals, e.g., 0.02 for 2%)
            current_prices: Current prices at each prediction point

        Returns:
            Predicted future prices
        """
        # future_price = current_price * (1 + return)
        predicted_prices = current_prices * (1 + predicted_returns)
        return predicted_prices

    def run_backtest_from_returns(
        self,
        predicted_returns: np.ndarray,
        actual_returns: np.ndarray,
        current_prices: np.ndarray,
        timestamps: pd.DatetimeIndex,
        denormalize: bool = True
    ) -> 'BacktestResults':
        """
        Run backtest using return predictions (from target_return model).

        This is the preferred method when using a model trained on target_return.

        Args:
            predicted_returns: (n_samples,) predicted future returns (normalized)
            actual_returns: (n_samples,) actual future returns (normalized)
            current_prices: (n_samples,) current prices in USD
            timestamps: (n_samples,) timestamps
            denormalize: Whether to denormalize returns using target_scaler

        Returns:
            BacktestResults with all metrics
        """
        if self.verbose:
            print(f"\n[BACKTEST] Running backtest from return predictions...")
            print(f"[BACKTEST]   - Samples: {len(predicted_returns)}")

        # Denormalize returns if needed
        if denormalize:
            predicted_returns = self.denormalize_returns(predicted_returns)
            actual_returns = self.denormalize_returns(actual_returns)

        if self.verbose:
            print(f"[BACKTEST]   - Predicted return range: [{predicted_returns.min():.4f}, {predicted_returns.max():.4f}]")
            print(f"[BACKTEST]   - Actual return range: [{actual_returns.min():.4f}, {actual_returns.max():.4f}]")

        # Convert returns to prices
        predicted_prices = self.returns_to_prices(predicted_returns, current_prices)
        actual_prices = self.returns_to_prices(actual_returns, current_prices)

        if self.verbose:
            print(f"[BACKTEST]   - Converted to price predictions")

        # Run standard backtest with prices
        return self.run_backtest(
            predictions=predicted_prices,
            actual_prices=actual_prices,
            timestamps=timestamps,
            initial_prices=current_prices
        )

    def run_backtest(
        self,
        predictions: np.ndarray,
        actual_prices: np.ndarray,
        timestamps: pd.DatetimeIndex,
        initial_prices: Optional[np.ndarray] = None
    ) -> BacktestResults:
        """
        Run backtest on predictions

        IMPORTANT: All price arrays must be DENORMALIZED (real prices in USD),
        not normalized values from the scaler.

        Args:
            predictions: (n_samples,) predicted future prices in USD (denormalized)
            actual_prices: (n_samples,) actual future prices in USD (denormalized)
            timestamps: (n_samples,) timestamps
            initial_prices: (n_samples,) current prices before prediction in USD (denormalized)

        Returns:
            BacktestResults with all metrics
        """
        if self.verbose:
            print(f"\n[BACKTEST] Running backtest...")
            print(f"[BACKTEST]   - Samples: {len(predictions)}")
            # Handle both numpy arrays and pandas Series
            ts_first = timestamps.iloc[0] if hasattr(timestamps, 'iloc') else timestamps[0]
            ts_last = timestamps.iloc[-1] if hasattr(timestamps, 'iloc') else timestamps[-1]
            print(f"[BACKTEST]   - Period: {ts_first} to {ts_last}")

        # If initial prices not provided, shift actual prices by 1
        if initial_prices is None:
            initial_prices = np.roll(actual_prices, 1)
            initial_prices[0] = actual_prices[0]

        # Initialize portfolio
        cash = self.config.initial_capital
        position = 0.0  # Amount of crypto held
        trades = []
        equity_values = []

        # Track for buy-and-hold comparison
        initial_price = initial_prices[0]
        buy_hold_position = cash / initial_price  # Buy all at start

        # Debug: Print first few samples to understand the data
        trade_signals = {'buy': 0, 'sell': 0, 'hold': 0}

        if self.verbose and len(predictions) > 0:
            print(f"\n[BACKTEST] Sample data (first 10):")
            for i in range(min(10, len(predictions))):
                change_pct = (predictions[i] - initial_prices[i]) / (initial_prices[i] + 1e-8) * 100

                # Determine what signal this would be
                if change_pct > self.config.confidence_threshold * 100:
                    signal = "BUY [BUY]"
                    trade_signals['buy'] += 1
                elif change_pct < -self.config.confidence_threshold * 100:
                    signal = "SELL [SELL]"
                    trade_signals['sell'] += 1
                else:
                    signal = "HOLD [HOLD]"
                    trade_signals['hold'] += 1

                print(f"  {i}: current=${initial_prices[i]:.2f}, predicted=${predictions[i]:.2f}, actual=${actual_prices[i]:.2f}")
                print(f"      -> change: {change_pct:+.3f}% (threshold: +/-{self.config.confidence_threshold*100:.2f}%) -> {signal}")

            print(f"\n[BACKTEST] Potential signals in all data:")
            print(f"  BUY signals:  {trade_signals['buy']}")
            print(f"  SELL signals: {trade_signals['sell']}")
            print(f"  HOLD signals: {trade_signals['hold']}")

        for i in range(len(predictions)):
            current_price = initial_prices[i]
            predicted_next = predictions[i]
            actual_next = actual_prices[i]
            timestamp = timestamps[i]

            # Calculate predicted price change
            price_change_pct = (predicted_next - current_price) / (current_price + 1e-8)

            # Decide action based on prediction
            action = 'HOLD'
            trade_amount = 0.0
            trade_value = 0.0

            # Current portfolio value
            portfolio_value = cash + (position * current_price)

            # BUY signal: predicted to go up significantly
            if price_change_pct > self.config.confidence_threshold and cash > self.config.min_trade_amount:
                # Buy with available cash (keep small reserve for fees)
                trade_value = cash * 0.95
                fee = trade_value * self.config.trade_fee
                trade_amount = (trade_value - fee) / current_price

                position += trade_amount
                cash -= trade_value
                action = 'BUY'

            # SELL signal: predicted to go down significantly
            elif price_change_pct < -self.config.confidence_threshold and position > 0:
                # Sell all position
                trade_amount = position
                trade_value = trade_amount * current_price
                fee = trade_value * self.config.trade_fee

                cash += (trade_value - fee)
                position = 0.0
                action = 'SELL'

            # Record trade
            portfolio_value_after = cash + (position * current_price)

            trade = Trade(
                timestamp=timestamp,
                action=action,
                price=current_price,
                amount=trade_amount,
                value=trade_value,
                balance_after=portfolio_value_after,
                prediction=predicted_next,
                actual_next=actual_next
            )
            trades.append(trade)
            equity_values.append(portfolio_value_after)

        # Final portfolio value
        final_price = actual_prices[-1]
        final_value = cash + (position * final_price)

        # Buy and hold final value
        buy_hold_final = buy_hold_position * final_price

        # Calculate metrics
        results = self._calculate_metrics(
            trades=trades,
            equity_values=equity_values,
            timestamps=timestamps,
            predictions=predictions,
            actual_prices=actual_prices,
            initial_capital=self.config.initial_capital,
            final_value=final_value,
            buy_hold_final=buy_hold_final
        )

        if self.verbose:
            self._print_results(results)

        return results

    def run_backtest_with_strategy(
        self,
        strategy: 'BaseStrategy',
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> 'BacktestResults':
        """
        Run backtest using a pluggable trading strategy.

        This method uses a strategy object to generate buy/sell signals
        from the data, then simulates trading based on those signals.

        Args:
            strategy: Strategy instance (implements BaseStrategy)
            data: DataFrame with OHLCV and technical indicators
            price_column: Column name for prices (default: 'close')

        Returns:
            BacktestResults with all metrics
        """
        from src.ml.strategies.base import BaseStrategy, TradeSignal

        if self.verbose:
            print(f"\n[BACKTEST] Running strategy backtest...")
            print(f"[BACKTEST]   - Strategy: {strategy.name}")
            print(f"[BACKTEST]   - Description: {strategy.description}")
            print(f"[BACKTEST]   - Data samples: {len(data)}")

        # Validate data has required features
        is_valid, missing = strategy.validate_data(data)
        if not is_valid:
            raise ValueError(f"Data missing required features for strategy '{strategy.name}': {missing}")

        # Generate signals from strategy
        signals = strategy.generate_signals(data)

        if self.verbose:
            buy_count = sum(1 for s in signals if s.action == 'BUY')
            sell_count = sum(1 for s in signals if s.action == 'SELL')
            hold_count = sum(1 for s in signals if s.action == 'HOLD')
            print(f"[BACKTEST]   - Signals generated: {len(signals)}")
            print(f"[BACKTEST]   - BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")

        # Initialize portfolio
        cash = self.config.initial_capital
        position = 0.0
        trades = []
        equity_values = []

        # Get prices and timestamps
        prices = data[price_column].values
        timestamps = data['datetime'] if 'datetime' in data.columns else pd.RangeIndex(len(data))

        # Track for buy-and-hold comparison
        initial_price = prices[0]
        buy_hold_position = cash / initial_price

        # Cooldown tracking
        min_confidence = strategy.config.min_confidence if hasattr(strategy, 'config') else 0.5
        cooldown = 0

        # Execute trades based on signals
        for i, signal in enumerate(signals):
            current_price = prices[i]
            timestamp = timestamps.iloc[i] if hasattr(timestamps, 'iloc') else timestamps[i]

            # Current portfolio value
            portfolio_value = cash + (position * current_price)

            action = 'HOLD'
            trade_amount = 0.0
            trade_value = 0.0

            # Cooldown period
            if cooldown > 0:
                cooldown -= 1
            else:
                # Check if signal meets confidence threshold
                if signal.confidence >= min_confidence:
                    if signal.action == 'BUY' and cash > self.config.min_trade_amount:
                        # Buy with available cash
                        trade_value = cash * strategy.config.position_size if hasattr(strategy, 'config') else cash * 0.95
                        fee = trade_value * self.config.trade_fee
                        trade_amount = (trade_value - fee) / current_price

                        position += trade_amount
                        cash -= trade_value
                        action = 'BUY'
                        cooldown = strategy.config.cooldown_periods if hasattr(strategy, 'config') else 1

                    elif signal.action == 'SELL' and position > 0:
                        # Sell all position
                        trade_amount = position
                        trade_value = trade_amount * current_price
                        fee = trade_value * self.config.trade_fee

                        cash += (trade_value - fee)
                        position = 0.0
                        action = 'SELL'
                        cooldown = strategy.config.cooldown_periods if hasattr(strategy, 'config') else 1

            # Record trade
            portfolio_value_after = cash + (position * current_price)

            # Get actual next price for prediction accuracy (if available)
            actual_next = prices[i + 1] if i < len(prices) - 1 else current_price

            trade = Trade(
                timestamp=timestamp,
                action=action,
                price=current_price,
                amount=trade_amount,
                value=trade_value,
                balance_after=portfolio_value_after,
                prediction=signal.price,
                actual_next=actual_next
            )
            trades.append(trade)
            equity_values.append(portfolio_value_after)

        # Final portfolio value
        final_price = prices[-1]
        final_value = cash + (position * final_price)

        # Buy and hold final value
        buy_hold_final = buy_hold_position * final_price

        # For strategy backtest, prediction accuracy metrics are based on signals vs actual moves
        predictions = np.array([s.price for s in signals])
        actual_prices = prices

        # Calculate metrics
        results = self._calculate_metrics(
            trades=trades,
            equity_values=equity_values,
            timestamps=timestamps,
            predictions=predictions,
            actual_prices=actual_prices,
            initial_capital=self.config.initial_capital,
            final_value=final_value,
            buy_hold_final=buy_hold_final
        )

        if self.verbose:
            print(f"\n[BACKTEST] Strategy: {strategy.name}")
            self._print_results(results)

        return results

    def compare_strategies(
        self,
        strategies: List['BaseStrategy'],
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> Dict[str, 'BacktestResults']:
        """
        Compare multiple strategies on the same data.

        Args:
            strategies: List of strategy instances
            data: DataFrame with OHLCV and indicators
            price_column: Column name for prices

        Returns:
            Dict mapping strategy name to BacktestResults
        """
        results = {}

        if self.verbose:
            print(f"\n[BACKTEST] Comparing {len(strategies)} strategies...")

        for strategy in strategies:
            try:
                result = self.run_backtest_with_strategy(strategy, data, price_column)
                results[strategy.name] = result
            except Exception as e:
                print(f"[BACKTEST] Error running strategy '{strategy.name}': {e}")
                continue

        # Print comparison summary
        if self.verbose and results:
            print(f"\n{'='*80}")
            print(f"STRATEGY COMPARISON SUMMARY")
            print(f"{'='*80}")
            print(f"{'Strategy':<20} {'Return %':>12} {'Win Rate':>12} {'Trades':>10} {'vs B&H':>12}")
            print(f"{'-'*80}")

            for name, res in sorted(results.items(), key=lambda x: x[1].total_return_pct, reverse=True):
                print(f"{name:<20} {res.total_return_pct:>+11.2f}% {res.win_rate:>11.1f}% "
                      f"{res.num_trades:>10} {res.excess_return:>+11.2f}")

            print(f"{'='*80}")

        return results

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_values: List[float],
        timestamps: pd.DatetimeIndex,
        predictions: np.ndarray,
        actual_prices: np.ndarray,
        initial_capital: float,
        final_value: float,
        buy_hold_final: float
    ) -> BacktestResults:
        """Calculate all backtest metrics including ICE and Sharpe ratio"""

        # Portfolio performance
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100

        # Buy and hold performance
        buy_hold_return = buy_hold_final - initial_capital
        buy_hold_return_pct = (buy_hold_return / initial_capital) * 100
        excess_return = total_return - buy_hold_return

        # Trade statistics
        buy_trades = [t for t in trades if t.action == 'BUY']
        sell_trades = [t for t in trades if t.action == 'SELL']
        num_trades = len(buy_trades) + len(sell_trades)

        # Calculate wins/losses by comparing entry and exit prices
        wins = 0
        losses = 0
        total_profit = 0.0
        total_loss = 0.0

        # Match buy/sell pairs
        for i, buy in enumerate(buy_trades):
            # Find next sell after this buy
            sell = next((s for s in sell_trades if s.timestamp > buy.timestamp), None)
            if sell:
                profit_pct = (sell.price - buy.price) / buy.price
                if profit_pct > 0:
                    wins += 1
                    total_profit += profit_pct
                else:
                    losses += 1
                    total_loss += abs(profit_pct)

        win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0.0

        # Profit/Loss ratio
        avg_profit = total_profit / wins if wins > 0 else 0.0
        avg_loss = total_loss / losses if losses > 0 else 1e-8
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0

        # Prediction accuracy metrics
        mae = np.mean(np.abs(predictions - actual_prices))
        rmse = np.sqrt(np.mean((predictions - actual_prices) ** 2))

        # R2 score
        ss_res = np.sum((actual_prices - predictions) ** 2)
        ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # Direction accuracy (using safe calculation to prevent NaN)
        dir_acc = safe_direction_accuracy(predictions, actual_prices)

        # Equity curve
        equity_curve = pd.Series(equity_values, index=timestamps)
        equity_array = np.array(equity_values)

        # ===== NEW: ICE and Sharpe Ratio Calculations =====

        # Calculate returns from equity curve
        if len(equity_array) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            returns = returns[np.isfinite(returns)]
        else:
            returns = np.array([0.0])

        # Information Coefficient (correlation between predictions and actuals)
        # Using returns rather than prices for better IC interpretation
        if len(predictions) > 2 and len(actual_prices) > 2:
            pred_returns = np.diff(predictions) / predictions[:-1]
            actual_returns = np.diff(actual_prices) / actual_prices[:-1]
            ic = information_coefficient(actual_returns, pred_returns)
        else:
            ic = 0.0

        # Sharpe Ratio (risk-adjusted return)
        sharpe = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=8760)

        # Sortino Ratio (downside risk-adjusted return)
        sortino = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=8760)

        # Maximum Drawdown
        max_dd, _, _ = max_drawdown(equity_array)
        max_dd_pct = max_dd * 100

        # Calmar Ratio (return / max drawdown)
        calmar = calmar_ratio(returns, equity_array, periods_per_year=8760)

        # Volatility (annualized)
        volatility = np.std(returns, ddof=1) * np.sqrt(8760) if len(returns) > 1 else 0.0

        if self.verbose:
            print(f"\n[BACKTEST] Risk-Adjusted Metrics:")
            print(f"[BACKTEST]   - Information Coefficient (IC): {ic:.4f}")
            print(f"[BACKTEST]   - Sharpe Ratio: {sharpe:.4f}")
            print(f"[BACKTEST]   - Sortino Ratio: {sortino:.4f}")
            print(f"[BACKTEST]   - Max Drawdown: {max_dd_pct:.2f}%")
            print(f"[BACKTEST]   - Calmar Ratio: {calmar:.4f}")
            print(f"[BACKTEST]   - Annualized Volatility: {volatility * 100:.2f}%")

        return BacktestResults(
            initial_capital=initial_capital,
            final_capital=final_value,
            total_return=total_return,
            total_return_pct=total_return_pct,
            num_trades=num_trades,
            num_wins=wins,
            num_losses=losses,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            prediction_mae=mae,
            prediction_rmse=rmse,
            prediction_r2=r2,
            direction_accuracy=dir_acc,
            # NEW: Risk-adjusted metrics
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            information_coefficient=ic,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=calmar,
            volatility=volatility,
            # Trade history
            trades=trades,
            equity_curve=equity_curve,
            buy_hold_return=buy_hold_return,
            buy_hold_return_pct=buy_hold_return_pct,
            excess_return=excess_return
        )

    def _print_results(self, results: BacktestResults):
        """Print backtest results"""
        print(f"\n{'='*80}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*80}")

        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"  Initial Capital:    ${results.initial_capital:,.2f}")
        print(f"  Final Capital:      ${results.final_capital:,.2f}")
        print(f"  Total Return:       ${results.total_return:+,.2f} ({results.total_return_pct:+.2f}%)")

        print(f"\nBUY & HOLD COMPARISON:")
        print(f"  Buy & Hold Return:  ${results.buy_hold_return:+,.2f} ({results.buy_hold_return_pct:+.2f}%)")
        print(f"  Excess Return:      ${results.excess_return:+,.2f}")
        print(f"  Strategy {'OUTPERFORMED' if results.excess_return > 0 else 'UNDERPERFORMED'} buy & hold")

        print(f"\nTRADING STATISTICS:")
        print(f"  Total Trades:       {results.num_trades}")
        print(f"  Winning Trades:     {results.num_wins}")
        print(f"  Losing Trades:      {results.num_losses}")
        print(f"  Win Rate:           {results.win_rate:.2f}%")
        print(f"  Profit/Loss Ratio:  {results.profit_loss_ratio:.2f}")

        print(f"\nRISK-ADJUSTED METRICS (ICE & SHARPE FOCUS):")
        print(f"  Information Coef:   {results.information_coefficient:.4f}")
        print(f"  Sharpe Ratio:       {results.sharpe_ratio:.4f}")
        print(f"  Sortino Ratio:      {results.sortino_ratio:.4f}")
        print(f"  Max Drawdown:       {results.max_drawdown_pct:.2f}%")
        print(f"  Calmar Ratio:       {results.calmar_ratio:.4f}")
        print(f"  Volatility (Ann.):  {results.volatility * 100:.2f}%")

        # Interpret ICE with detailed ratings
        ic = results.information_coefficient
        if ic > 0.15:
            ic_rating = "EXCELLENT"
            ic_desc = "IC > 0.15: Exceptional predictive power (rare)"
        elif ic > 0.10:
            ic_rating = "STRONG"
            ic_desc = "IC > 0.10: Strong predictive power"
        elif ic > 0.05:
            ic_rating = "GOOD"
            ic_desc = "IC > 0.05: Meaningful predictive power"
        elif ic > 0.02:
            ic_rating = "MODERATE"
            ic_desc = "IC > 0.02: Some predictive signal"
        elif ic > 0:
            ic_rating = "WEAK"
            ic_desc = "IC > 0.00: Marginal predictive power"
        elif ic > -0.02:
            ic_rating = "NONE"
            ic_desc = "IC ~ 0.00: No predictive power"
        else:
            ic_rating = "INVERSE"
            ic_desc = "IC < -0.02: Predictions inversely correlated (bad)"
        print(f"  IC Rating:          {ic_rating} ({ic_desc})")

        # Interpret Sharpe with detailed ratings
        sharpe = results.sharpe_ratio
        if sharpe > 3:
            sharpe_rating = "EXCEPTIONAL"
            sharpe_desc = "Sharpe > 3: Outstanding (check for overfitting)"
        elif sharpe > 2:
            sharpe_rating = "EXCELLENT"
            sharpe_desc = "Sharpe > 2: Very good risk-adjusted returns"
        elif sharpe > 1:
            sharpe_rating = "GOOD"
            sharpe_desc = "Sharpe > 1: Good risk-adjusted returns"
        elif sharpe > 0.5:
            sharpe_rating = "ACCEPTABLE"
            sharpe_desc = "Sharpe > 0.5: Acceptable but could improve"
        elif sharpe > 0:
            sharpe_rating = "MARGINAL"
            sharpe_desc = "Sharpe > 0: Positive but suboptimal"
        else:
            sharpe_rating = "POOR"
            sharpe_desc = "Sharpe < 0: Negative risk-adjusted returns"
        print(f"  Sharpe Rating:      {sharpe_rating} ({sharpe_desc})")

        print(f"\nPREDICTION ACCURACY:")
        print(f"  MAE:                {results.prediction_mae:.4f}")
        print(f"  RMSE:               {results.prediction_rmse:.4f}")
        print(f"  R2 Score:           {results.prediction_r2:.4f}")
        print(f"  Direction Accuracy: {results.direction_accuracy:.2f}%")

        print(f"\n{'='*80}")

    def plot_results(self, results: BacktestResults, save_path: Optional[str] = None):
        """
        Create visualization plots for backtest results

        Args:
            results: BacktestResults object
            save_path: Optional path to save plot
        """
        # Import Figure to create figure without using plt global state
        from matplotlib.figure import Figure

        # Create new figure with explicit size (not using plt.figure)
        fig = Figure(figsize=(15, 10))
        fig.suptitle('Backtesting Results', fontsize=16, fontweight='bold')

        # Create subplots manually
        axes = []
        axes.append(fig.add_subplot(2, 2, 1))
        axes.append(fig.add_subplot(2, 2, 2))
        axes.append(fig.add_subplot(2, 2, 3))
        axes.append(fig.add_subplot(2, 2, 4))
        axes = np.array(axes).reshape(2, 2)

        # Plot 1: Equity Curve
        ax1 = axes[0, 0]
        ax1.plot(results.equity_curve.index, results.equity_curve.values,
                linewidth=2, color='#2196F3', label='Strategy')
        ax1.axhline(y=results.initial_capital, color='gray', linestyle='--',
                   label='Initial Capital', alpha=0.5)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Returns Comparison
        ax2 = axes[0, 1]
        returns_data = [
            results.total_return_pct,
            results.buy_hold_return_pct
        ]
        colors = ['#4CAF50' if r > 0 else '#F44336' for r in returns_data]
        bars = ax2.bar(['Strategy', 'Buy & Hold'], returns_data, color=colors)
        ax2.set_title('Total Return Comparison')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top')

        # Plot 3: Win/Loss Statistics
        ax3 = axes[1, 0]
        if results.num_wins + results.num_losses > 0:
            trade_counts = [results.num_wins, results.num_losses]
            colors_trades = ['#4CAF50', '#F44336']
            ax3.pie(trade_counts, labels=['Wins', 'Losses'], colors=colors_trades,
                   autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'Win Rate: {results.win_rate:.1f}% | P/L Ratio: {results.profit_loss_ratio:.2f}')
        else:
            ax3.text(0.5, 0.5, 'No trades executed',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Win/Loss Statistics')

        # Plot 4: Performance Metrics
        ax4 = axes[1, 1]
        metrics_labels = ['MAE', 'RMSE', 'R2', 'Dir Acc']
        metrics_values = [
            results.prediction_mae,
            results.prediction_rmse,
            results.prediction_r2,
            results.direction_accuracy / 100  # Scale to 0-1
        ]

        # Normalize for comparison
        max_val = max(metrics_values[:2])  # MAE and RMSE scale
        normalized_values = [
            metrics_values[0] / max_val if max_val > 0 else 0,
            metrics_values[1] / max_val if max_val > 0 else 0,
            metrics_values[2],  # R2 already 0-1
            metrics_values[3]   # Direction accuracy 0-1
        ]

        bars = ax4.barh(metrics_labels, normalized_values, color='#FF9800')
        ax4.set_title('Prediction Performance (Normalized)')
        ax4.set_xlabel('Score')
        ax4.set_xlim(0, 1)
        ax4.grid(True, alpha=0.3, axis='x')

        # Add actual values as text
        for i, (bar, val) in enumerate(zip(bars, metrics_values)):
            if metrics_labels[i] == 'Dir Acc':
                text = f'{val*100:.1f}%'
            else:
                text = f'{val:.4f}'
            ax4.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f' {text}', va='center')

        try:
            fig.tight_layout()
        except Exception as e:
            # Tight layout can fail sometimes, just warn and continue
            print(f"[BACKTEST] Warning: tight_layout failed: {e}")

        if save_path:
            try:
                # Need a canvas to save the figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                canvas = FigureCanvasAgg(fig)
                canvas.print_figure(save_path, dpi=150, bbox_inches='tight')
                print(f"[BACKTEST] Plot saved to {save_path}")
            except Exception as e:
                print(f"[BACKTEST] Error saving plot: {e}")
                # Try without tight bbox as fallback
                try:
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    canvas = FigureCanvasAgg(fig)
                    canvas.print_figure(save_path, dpi=150)
                    print(f"[BACKTEST] Plot saved to {save_path} (without tight bbox)")
                except Exception as e2:
                    print(f"[BACKTEST] Failed to save plot: {e2}")

        return fig


@dataclass
class WalkForwardResults:
    """Results from walk-forward backtesting"""
    num_folds: int
    fold_results: List[BacktestResults]

    # Aggregated metrics
    avg_sharpe_ratio: float
    avg_sortino_ratio: float
    avg_information_coefficient: float
    avg_total_return_pct: float
    avg_win_rate: float
    avg_direction_accuracy: float
    avg_max_drawdown_pct: float

    # Stability metrics
    sharpe_std: float
    ic_std: float
    return_std: float

    # Overall performance
    cumulative_return_pct: float
    worst_fold_return_pct: float
    best_fold_return_pct: float


class WalkForwardBacktester:
    """
    Walk-Forward Backtesting for realistic model evaluation.

    Walk-forward testing simulates real trading by:
    1. Training on historical data up to time T
    2. Testing on T to T+N (out-of-sample)
    3. Moving forward and retraining periodically

    This provides more realistic performance estimates than
    a single train/test split.
    """

    def __init__(
        self,
        n_folds: int = 5,
        train_ratio: float = 0.7,
        gap_periods: int = 24,  # Gap between train and test to avoid leakage
        retrain_every: int = None,  # Periods between retraining (None = per fold)
        verbose: bool = True
    ):
        """
        Initialize walk-forward backtester.

        Args:
            n_folds: Number of walk-forward folds
            train_ratio: Ratio of each fold used for training
            gap_periods: Gap between train/test to avoid leakage (hours)
            retrain_every: Retrain model every N periods (None = each fold)
            verbose: Print progress
        """
        self.n_folds = n_folds
        self.train_ratio = train_ratio
        self.gap_periods = gap_periods
        self.retrain_every = retrain_every
        self.verbose = verbose

        if self.verbose:
            print(f"\n[WALK-FORWARD] Initializing Walk-Forward Backtester")
            print(f"[WALK-FORWARD]   - Folds: {n_folds}")
            print(f"[WALK-FORWARD]   - Train ratio: {train_ratio:.1%}")
            print(f"[WALK-FORWARD]   - Gap periods: {gap_periods}")

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        model_factory,
        feature_columns: List[str],
        target_column: str = 'target_return',
        sequence_length: int = 168,
        backtest_config: Optional[BacktestConfig] = None
    ) -> WalkForwardResults:
        """
        Run walk-forward backtesting.

        Args:
            data: Full DataFrame with features and target
            model_factory: Callable that creates and trains a model
                          Signature: model_factory(train_data) -> trained_model
            feature_columns: List of feature column names
            target_column: Target column name
            sequence_length: Sequence length for model
            backtest_config: Backtest configuration

        Returns:
            WalkForwardResults with all fold results and aggregated metrics
        """
        if self.verbose:
            print(f"\n[WALK-FORWARD] Starting walk-forward backtest...")
            print(f"[WALK-FORWARD]   - Total samples: {len(data)}")
            print(f"[WALK-FORWARD]   - Features: {len(feature_columns)}")

        # Calculate fold sizes
        total_samples = len(data)
        fold_size = total_samples // self.n_folds

        if fold_size < sequence_length * 3:
            raise ValueError(f"Insufficient data for {self.n_folds} folds. "
                           f"Fold size {fold_size} < min required {sequence_length * 3}")

        fold_results = []
        backtester = CryptoBacktester(config=backtest_config or BacktestConfig(), verbose=False)

        for fold_idx in range(self.n_folds):
            if self.verbose:
                print(f"\n[WALK-FORWARD] === Fold {fold_idx + 1}/{self.n_folds} ===")

            # Calculate indices for this fold
            # Expanding window: train on all data up to test start
            test_start = (fold_idx + 1) * fold_size
            test_end = min(test_start + fold_size, total_samples)
            train_end = test_start - self.gap_periods

            if train_end < sequence_length * 2:
                if self.verbose:
                    print(f"[WALK-FORWARD] Skipping fold {fold_idx + 1}: insufficient training data")
                continue

            # Split data
            train_data = data.iloc[:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()

            if self.verbose:
                print(f"[WALK-FORWARD]   - Train: 0 to {train_end} ({len(train_data)} samples)")
                print(f"[WALK-FORWARD]   - Test: {test_start} to {test_end} ({len(test_data)} samples)")

            try:
                # Train model on this fold's training data
                model = model_factory(train_data)

                # Generate predictions on test data
                predictions = self._predict_with_model(
                    model, test_data, feature_columns, sequence_length
                )

                # Get actuals and prices
                actuals = test_data[target_column].values[sequence_length:]

                if 'close' in test_data.columns:
                    prices = test_data['close'].values[sequence_length:]
                else:
                    prices = np.ones(len(actuals)) * 100

                if 'datetime' in test_data.columns:
                    timestamps = test_data['datetime'].iloc[sequence_length:].reset_index(drop=True)
                else:
                    timestamps = pd.date_range(start='2024-01-01', periods=len(actuals), freq='h')

                # Truncate to match prediction length
                min_len = min(len(predictions), len(actuals), len(prices))
                predictions = predictions[:min_len]
                actuals = actuals[:min_len]
                prices = prices[:min_len]
                timestamps = timestamps[:min_len]

                # Run backtest for this fold
                if len(predictions) > 10:
                    fold_result = backtester.run_backtest_from_returns(
                        predicted_returns=predictions,
                        actual_returns=actuals,
                        current_prices=prices,
                        timestamps=timestamps,
                        denormalize=False  # Already in return space
                    )
                    fold_results.append(fold_result)

                    if self.verbose:
                        print(f"[WALK-FORWARD]   - Sharpe: {fold_result.sharpe_ratio:.4f}")
                        print(f"[WALK-FORWARD]   - IC: {fold_result.information_coefficient:.4f}")
                        print(f"[WALK-FORWARD]   - Return: {fold_result.total_return_pct:.2f}%")

            except Exception as e:
                if self.verbose:
                    print(f"[WALK-FORWARD] Error in fold {fold_idx + 1}: {e}")
                continue

        if not fold_results:
            raise ValueError("No successful folds completed")

        # Aggregate results
        results = self._aggregate_results(fold_results)

        if self.verbose:
            self._print_summary(results)

        return results

    def _predict_with_model(
        self,
        model,
        data: pd.DataFrame,
        feature_columns: List[str],
        sequence_length: int
    ) -> np.ndarray:
        """Generate predictions using the model."""
        import torch

        features = data[feature_columns].values.astype(np.float32)
        predictions = []

        model.eval()
        with torch.no_grad():
            for i in range(sequence_length, len(features)):
                seq = features[i - sequence_length:i]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)

                # Handle both classification and regression outputs
                output = model(seq_tensor)
                if isinstance(output, tuple):
                    # Classification model: (class_logits, regression_out)
                    _, reg_out = output
                    predictions.append(reg_out.numpy().flatten()[0])
                else:
                    predictions.append(output.numpy().flatten()[0])

        return np.array(predictions)

    def _aggregate_results(self, fold_results: List[BacktestResults]) -> WalkForwardResults:
        """Aggregate results across all folds."""
        sharpes = [r.sharpe_ratio for r in fold_results]
        sortinos = [r.sortino_ratio for r in fold_results]
        ics = [r.information_coefficient for r in fold_results]
        returns = [r.total_return_pct for r in fold_results]
        win_rates = [r.win_rate for r in fold_results]
        dir_accs = [r.direction_accuracy for r in fold_results]
        drawdowns = [r.max_drawdown_pct for r in fold_results]

        # Calculate cumulative return (compounding)
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r / 100)
        cumulative_return_pct = (cumulative - 1) * 100

        return WalkForwardResults(
            num_folds=len(fold_results),
            fold_results=fold_results,
            avg_sharpe_ratio=np.mean(sharpes),
            avg_sortino_ratio=np.mean(sortinos),
            avg_information_coefficient=np.mean(ics),
            avg_total_return_pct=np.mean(returns),
            avg_win_rate=np.mean(win_rates),
            avg_direction_accuracy=np.mean(dir_accs),
            avg_max_drawdown_pct=np.mean(drawdowns),
            sharpe_std=np.std(sharpes),
            ic_std=np.std(ics),
            return_std=np.std(returns),
            cumulative_return_pct=cumulative_return_pct,
            worst_fold_return_pct=min(returns),
            best_fold_return_pct=max(returns)
        )

    def _print_summary(self, results: WalkForwardResults):
        """Print walk-forward summary."""
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD BACKTEST RESULTS ({results.num_folds} folds)")
        print(f"{'='*80}")

        print(f"\nRISK-ADJUSTED METRICS (avg +/- std):")
        print(f"  Sharpe Ratio:      {results.avg_sharpe_ratio:+.4f} +/- {results.sharpe_std:.4f}")
        print(f"  Sortino Ratio:     {results.avg_sortino_ratio:+.4f}")
        print(f"  Information Coef:  {results.avg_information_coefficient:+.4f} +/- {results.ic_std:.4f}")

        print(f"\nRETURN METRICS:")
        print(f"  Avg Fold Return:   {results.avg_total_return_pct:+.2f}% +/- {results.return_std:.2f}%")
        print(f"  Cumulative Return: {results.cumulative_return_pct:+.2f}%")
        print(f"  Best Fold:         {results.best_fold_return_pct:+.2f}%")
        print(f"  Worst Fold:        {results.worst_fold_return_pct:+.2f}%")

        print(f"\nTRADING METRICS:")
        print(f"  Avg Win Rate:      {results.avg_win_rate:.1f}%")
        print(f"  Avg Direction Acc: {results.avg_direction_accuracy:.1f}%")
        print(f"  Avg Max Drawdown:  {results.avg_max_drawdown_pct:.2f}%")

        # Stability assessment
        sharpe_cv = results.sharpe_std / (abs(results.avg_sharpe_ratio) + 1e-8)
        if sharpe_cv < 0.5:
            stability = "STABLE"
        elif sharpe_cv < 1.0:
            stability = "MODERATE"
        else:
            stability = "UNSTABLE"

        print(f"\nSTABILITY ASSESSMENT:")
        print(f"  Sharpe CV:         {sharpe_cv:.2f} ({stability})")

        # Overall rating
        if results.avg_sharpe_ratio > 1.0 and results.avg_information_coefficient > 0.05:
            rating = "EXCELLENT - Ready for paper trading"
        elif results.avg_sharpe_ratio > 0.5 and results.avg_information_coefficient > 0.02:
            rating = "GOOD - Consider further optimization"
        elif results.avg_sharpe_ratio > 0:
            rating = "MARGINAL - Needs improvement"
        else:
            rating = "POOR - Do not deploy"

        print(f"  Overall Rating:    {rating}")
        print(f"{'='*80}")


if __name__ == "__main__":
    print("Cryptocurrency Backtesting Module")
    print("="*60)
    print("\nTest model predictions with realistic trading simulation")
