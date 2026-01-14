# Crypto AI Predictor - Improvement Recommendations

## Current Issues

### 1. Test Graph Shows Only 1 Sample
**Problem**: TimeSeriesDataSet with model parameters produces only 1 valid sequence
**Impact**: Graph shows single invisible point, can't evaluate model visually

**Solutions**:
- Use rolling window approach: slide window across test data to generate multiple predictions
- Reduce context_length for testing to get more samples
- Show prediction error distribution instead of raw predictions

### 2. Backtesting Produces No Trades
**Problem**: Model MAE of $119 on $90K = 0.13% change, below 1% confidence threshold
**Impact**: Can't evaluate trading strategy effectiveness

**Solutions**:
- **Lower confidence threshold**: Test with 0.1%, 0.25%, 0.5% thresholds
- **Multiple strategies**: Implement trend-following, mean-reversion, breakout strategies
- **Adaptive thresholds**: Adjust based on volatility (higher threshold in volatile markets)
- **Use prediction confidence**: Weight trades by model confidence

### 3. Model Predictions Too Conservative
**Problem**: Predictions very close to current price (low variance)
**Root Cause**:
- Model trained on MSE loss optimizes for mean prediction
- Regularization might be too strong
- Feature engineering might not capture enough price momentum

**Solutions**:
- **Use quantile loss**: Train on QuantileLoss instead of MSE to get prediction intervals
- **Add momentum features**: More emphasis on recent price trends
- **Ensemble predictions**: Combine multiple models with different lookback periods
- **Adjust learning rate**: Try higher learning rate for faster adaptation

## Recommended Feature Additions

### A. Enhanced Testing & Evaluation

```python
class EnhancedModelTester:
    """Comprehensive model testing with multiple metrics"""

    def rolling_window_test(self, test_data, window_size=24):
        """
        Test model using rolling window approach
        Generates predictions every hour across the test period
        """
        predictions = []
        actuals = []
        timestamps = []

        for i in range(len(test_data) - window_size - prediction_horizon):
            window = test_data[i:i+window_size]
            pred = model.predict(window)
            actual = test_data[i+window_size:i+window_size+prediction_horizon]

            predictions.append(pred)
            actuals.append(actual)
            timestamps.append(test_data.index[i+window_size])

        return predictions, actuals, timestamps

    def calculate_metrics(self, predictions, actuals):
        """
        Comprehensive metrics beyond MAE/RMSE
        """
        return {
            'mae': mean_absolute_error(actuals, predictions),
            'rmse': root_mean_squared_error(actuals, predictions),
            'mape': mean_absolute_percentage_error(actuals, predictions),
            'r2': r2_score(actuals, predictions),
            'directional_accuracy': (np.sign(np.diff(predictions)) == np.sign(np.diff(actuals))).mean(),
            'profit_correlation': correlation(predictions - actuals[:-1], actuals[1:] - actuals[:-1]),
            'sharpe_ratio': calculate_sharpe(predictions, actuals),
            'max_drawdown': calculate_max_drawdown(predictions, actuals)
        }
```

### B. Improved Backtesting System

```python
class AdvancedBacktester:
    """
    Multi-strategy backtesting with realistic constraints
    """

    def __init__(self, initial_capital=10000, max_position_size=0.2):
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.strategies = []

    def add_strategy(self, strategy):
        """Add trading strategy"""
        self.strategies.append(strategy)

    def run_backtest(self, predictions, prices, timestamps):
        """
        Run backtest with multiple strategies
        """
        results = {}

        for strategy in self.strategies:
            portfolio = self.simulate_strategy(strategy, predictions, prices)
            results[strategy.name] = {
                'final_value': portfolio.value,
                'return_pct': (portfolio.value - self.capital) / self.capital * 100,
                'sharpe': portfolio.sharpe_ratio,
                'max_drawdown': portfolio.max_drawdown,
                'win_rate': portfolio.win_rate,
                'num_trades': portfolio.num_trades,
                'avg_profit_per_trade': portfolio.avg_profit
            }

        return results

class TradingStrategy:
    """Base class for trading strategies"""

    def __init__(self, name):
        self.name = name

    def should_buy(self, prediction, current_price, confidence):
        raise NotImplementedError

    def should_sell(self, prediction, current_price, confidence):
        raise NotImplementedError

    def position_size(self, capital, confidence):
        """Calculate position size based on confidence"""
        base_size = capital * 0.1  # 10% of capital
        return base_size * confidence

class MomentumStrategy(TradingStrategy):
    """Buy on strong upward predictions, sell on strong downward"""

    def __init__(self, threshold=0.005):
        super().__init__("Momentum")
        self.threshold = threshold

    def should_buy(self, prediction, current_price, confidence):
        change_pct = (prediction - current_price) / current_price
        return change_pct > self.threshold and confidence > 0.6

    def should_sell(self, prediction, current_price, confidence):
        change_pct = (prediction - current_price) / current_price
        return change_pct < -self.threshold and confidence > 0.6

class MeanReversionStrategy(TradingStrategy):
    """Buy when predicted to revert up, sell when predicted to revert down"""

    def __init__(self, sma_period=24):
        super().__init__("MeanReversion")
        self.sma_period = sma_period

    def should_buy(self, prediction, current_price, sma, confidence):
        # Buy if currently below SMA but predicted to rise
        return current_price < sma and prediction > current_price and confidence > 0.6

    def should_sell(self, prediction, current_price, sma, confidence):
        # Sell if currently above SMA but predicted to fall
        return current_price > sma and prediction < current_price and confidence > 0.6
```

### C. Real-Time Model Monitoring

```python
class ModelMonitor:
    """Monitor model performance in real-time"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def add_prediction(self, pred, actual, timestamp):
        """Add new prediction-actual pair"""
        self.predictions.append(pred)
        self.actuals.append(actual)
        self.timestamps.append(timestamp)

    def get_current_metrics(self):
        """Calculate metrics on recent window"""
        if len(self.predictions) < 10:
            return None

        return {
            'mae': np.mean(np.abs(np.array(self.predictions) - np.array(self.actuals))),
            'rmse': np.sqrt(np.mean((np.array(self.predictions) - np.array(self.actuals))**2)),
            'direction_accuracy': self.calculate_directional_accuracy(),
            'drift_score': self.calculate_drift()
        }

    def calculate_drift(self):
        """
        Detect if model is drifting (performance degrading over time)
        """
        if len(self.predictions) < self.window_size:
            return 0

        # Compare first half vs second half error
        mid = len(self.predictions) // 2
        first_half_error = np.mean(np.abs(self.predictions[:mid] - self.actuals[:mid]))
        second_half_error = np.mean(np.abs(self.predictions[mid:] - self.actuals[mid:]))

        # Positive drift means model getting worse
        return (second_half_error - first_half_error) / first_half_error

    def alert_if_degraded(self, threshold=0.2):
        """Alert if model performance degraded by threshold %"""
        drift = self.calculate_drift()
        if drift > threshold:
            return f"⚠️ Model drift detected! Performance degraded by {drift*100:.1f}%"
        return None
```

### D. Enhanced UI Features

```python
# 1. Model Comparison Tab
class ModelComparisonPanel:
    """Compare multiple trained models side-by-side"""

    def compare_models(self, models, test_data):
        """
        Show:
        - Accuracy metrics comparison
        - Prediction plots overlaid
        - Trading performance comparison
        - Feature importance differences
        """
        pass

# 2. Prediction Confidence Display
class ConfidencePlot:
    """Show predictions with confidence intervals"""

    def plot_with_confidence(self, predictions, confidence_intervals):
        """
        Plot:
        - Central prediction line
        - 95% confidence band (shaded area)
        - 80% confidence band (darker shaded area)
        - Actual values for comparison
        """
        pass

# 3. Feature Importance Visualization
class FeatureExplainer:
    """Show which features contribute most to predictions"""

    def explain_prediction(self, model, input_data):
        """
        Use SHAP or LIME to show:
        - Top 10 most important features
        - Feature contribution to this specific prediction
        - Feature trends over time
        """
        pass

# 4. Live Trading Dashboard
class TradingDashboard:
    """Real-time trading performance"""

    def show_live_performance(self):
        """
        Display:
        - Current positions
        - P&L (realized + unrealized)
        - Win/loss ratio today
        - Strategy performance breakdown
        - Risk exposure
        """
        pass
```

### E. Risk Management System

```python
class RiskManager:
    """Comprehensive risk management"""

    def __init__(self, max_position_size=0.2, max_portfolio_risk=0.1):
        self.max_position_size = max_position_size  # Max 20% per position
        self.max_portfolio_risk = max_portfolio_risk  # Max 10% portfolio risk

    def calculate_position_size(self, capital, confidence, volatility):
        """
        Kelly Criterion adjusted by confidence
        """
        base_size = capital * self.max_position_size
        confidence_adj = base_size * confidence
        volatility_adj = confidence_adj * (1 / volatility)  # Reduce size in high volatility

        return min(volatility_adj, base_size)

    def should_allow_trade(self, portfolio, new_trade):
        """
        Check if trade passes risk checks
        """
        # Check position concentration
        if new_trade.value / portfolio.total_value > self.max_position_size:
            return False, "Position size too large"

        # Check portfolio risk
        total_risk = portfolio.calculate_risk()
        if total_risk > self.max_portfolio_risk:
            return False, "Portfolio risk limit exceeded"

        # Check correlation (don't add highly correlated positions)
        if portfolio.has_correlated_position(new_trade, threshold=0.8):
            return False, "Too correlated with existing positions"

        return True, "OK"

    def set_stop_loss_take_profit(self, entry_price, prediction, confidence):
        """
        Calculate optimal stop-loss and take-profit levels
        """
        predicted_change = (prediction - entry_price) / entry_price

        # Stop-loss: risk 0.5x the predicted gain
        stop_loss = entry_price * (1 - abs(predicted_change) * 0.5)

        # Take-profit: aim for 1.5x the predicted gain
        take_profit = entry_price * (1 + abs(predicted_change) * 1.5)

        # Adjust by confidence
        if confidence < 0.7:
            # Less confident: tighter stops
            stop_loss = entry_price * (1 - abs(predicted_change) * 0.3)
            take_profit = entry_price * (1 + abs(predicted_change) * 1.0)

        return stop_loss, take_profit
```

## Priority Implementation Order

### Phase 1: Fix Critical Issues (1-2 days)
1. ✅ Fix test graph to show multiple samples (rolling window)
2. ✅ Lower backtest confidence threshold to 0.25%
3. ✅ Add more debug logging for backtesting

### Phase 2: Enhanced Testing (2-3 days)
1. Implement rolling window test
2. Add comprehensive metrics (Sharpe, max drawdown, etc.)
3. Add prediction confidence intervals
4. Test on different market conditions

### Phase 3: Better Trading Strategies (3-4 days)
1. Implement multiple strategies (momentum, mean reversion)
2. Add adaptive confidence thresholds
3. Implement proper position sizing
4. Add risk management system

### Phase 4: UI Enhancements (3-4 days)
1. Model comparison panel
2. Real-time monitoring dashboard
3. Feature importance visualization
4. Live P&L tracking

### Phase 5: Production Readiness (2-3 days)
1. Add paper trading mode
2. Add model versioning
3. Implement automatic retraining
4. Add alerting system

## Immediate Quick Wins

### 1. Lower Backtest Threshold
```python
# In BacktestConfig
confidence_threshold: float = 0.0025  # 0.25% instead of 1%
```

### 2. Show More Test Samples
```python
# Use smaller context for testing
test_context_length = 24  # 1 day instead of full encoder length
```

### 3. Add Prediction Variance
```python
# Train with QuantileLoss to get uncertainty estimates
from pytorch_forecasting import QuantileLoss
loss = QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
```

## Testing on Real Data

### Test Scenarios
1. **Bull Market** (2020-2021): Test if model captures upward trends
2. **Bear Market** (2022): Test if model avoids losses
3. **Sideways** (2023): Test if model avoids overtrading
4. **High Volatility** (COVID crash, FTX collapse): Test risk management

### Performance Targets
- **Direction Accuracy**: >55% (better than random)
- **Sharpe Ratio**: >1.0 (better than buy-and-hold)
- **Max Drawdown**: <20%
- **Win Rate**: >45%
- **Profit Factor**: >1.5

## Conclusion

The application has a solid foundation but needs:
1. Better testing methodology (rolling windows)
2. More aggressive trading strategies (lower thresholds)
3. Enhanced risk management
4. Real-time monitoring
5. Better visualization of model uncertainty

Focus on Phase 1 quick wins first to make the system usable, then iterate on Phases 2-5 based on performance.
