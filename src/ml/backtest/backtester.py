"""
Backtesting Module for Cryptocurrency Trading Strategy
Tests model predictions on historical data with realistic trading simulation

IMPORTANT: This backtester works with RETURN predictions (target_return)
not raw price predictions. It converts predicted returns to price movements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
from src.ml.training.model_evaluator import safe_direction_accuracy


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

    # Trade history
    trades: List[Trade]
    equity_curve: pd.Series

    # Buy and hold comparison
    buy_hold_return: float
    buy_hold_return_pct: float
    excess_return: float


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
        """Calculate all backtest metrics"""

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
        direction_accuracy = safe_direction_accuracy(predictions, actual_prices)

        # Equity curve
        equity_curve = pd.Series(equity_values, index=timestamps)

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
            direction_accuracy=direction_accuracy,
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


if __name__ == "__main__":
    print("Cryptocurrency Backtesting Module")
    print("="*60)
    print("\nTest model predictions with realistic trading simulation")
