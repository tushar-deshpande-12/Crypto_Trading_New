"""
Capital-Aware Execution Module

Implements intelligent position sizing and trade execution based on:
- Sharpe ratio-based position sizing
- Minimum required move threshold
- Confidence scaling
- Volatility-adjusted edges
- Transparent decision logging

This module bridges ML predictions with actual trade execution,
ensuring capital preservation and risk management.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from src.utils.debug_logger import debug_log, get_debug_logger


@dataclass
class ExecutionConfig:
    """
    Configuration for capital-aware execution.

    All thresholds and multipliers can be tuned based on
    backtesting results and risk tolerance.
    """
    # Minimum required predicted move to consider trading (e.g., 0.028 = 2.8%)
    min_required_move: float = 0.02

    # Edge thresholds
    edge_min: float = 0.0015  # Minimum edge to trade (0.15%)

    # Fee assumptions
    trading_fee: float = 0.001  # 0.1% per trade

    # Sharpe-based position sizing
    target_sharpe: float = 2.0  # Target Sharpe for full position
    max_position: float = 1.0  # Maximum position size (100%)
    min_position: float = 0.05  # Minimum position size (5%)

    # Confidence and volatility adjustments
    confidence_scale: float = 0.7  # Scale factor for predicted edge
    volatility_mult: float = 2.0  # Volatility multiplier for edge clipping
    stop_mult: float = 1.1  # Stop loss multiplier

    # Volatility calculation window
    volatility_window: int = 50

    # Risk management
    max_drawdown_threshold: float = 0.15  # 15% max drawdown before reducing size
    kelly_fraction: float = 0.25  # Fraction of Kelly criterion to use

    # Transparency
    log_decisions: bool = True
    decision_log_path: str = "debug_logs/execution_decisions.json"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'min_required_move': self.min_required_move,
            'edge_min': self.edge_min,
            'trading_fee': self.trading_fee,
            'target_sharpe': self.target_sharpe,
            'max_position': self.max_position,
            'min_position': self.min_position,
            'confidence_scale': self.confidence_scale,
            'volatility_mult': self.volatility_mult,
            'stop_mult': self.stop_mult,
            'volatility_window': self.volatility_window,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'kelly_fraction': self.kelly_fraction,
        }


@dataclass
class TradeDecision:
    """
    Transparent trade decision with full reasoning.

    Every decision includes the rationale, confidence metrics,
    and all factors that influenced the decision.
    """
    timestamp: datetime
    symbol: str

    # Decision
    action: str  # 'BUY', 'SELL', 'HOLD'
    position_size: float  # 0.0 to 1.0

    # Price levels
    current_price: float
    predicted_price: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

    # Metrics that drove the decision
    predicted_return: float = 0.0
    predicted_move_pct: float = 0.0
    raw_edge: float = 0.0
    adjusted_edge: float = 0.0
    volatility: float = 0.0

    # Confidence and quality metrics
    confidence: float = 0.0
    information_coefficient: float = 0.0
    current_sharpe: float = 0.0
    risk_reward_ratio: float = 0.0

    # Decision reasoning (for transparency)
    reasoning: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Execution metadata
    executed: bool = False
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for logging."""
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol,
            'action': self.action,
            'position_size': self.position_size,
            'current_price': self.current_price,
            'predicted_price': self.predicted_price,
            'take_profit': self.take_profit,
            'stop_loss': self.stop_loss,
            'predicted_return': self.predicted_return,
            'predicted_move_pct': self.predicted_move_pct,
            'raw_edge': self.raw_edge,
            'adjusted_edge': self.adjusted_edge,
            'volatility': self.volatility,
            'confidence': self.confidence,
            'information_coefficient': self.information_coefficient,
            'current_sharpe': self.current_sharpe,
            'risk_reward_ratio': self.risk_reward_ratio,
            'reasoning': self.reasoning,
            'warnings': self.warnings,
            'executed': self.executed,
            'execution_price': self.execution_price,
            'execution_time': self.execution_time.isoformat() if self.execution_time else None,
        }

    def summary(self) -> str:
        """Generate human-readable summary of the decision."""
        lines = [
            f"=== Trade Decision: {self.symbol} ===",
            f"Time: {self.timestamp}",
            f"Action: {self.action}",
            f"Position Size: {self.position_size * 100:.1f}%",
            f"",
            f"Prices:",
            f"  Current: ${self.current_price:.2f}",
            f"  Predicted: ${self.predicted_price:.2f}",
        ]

        if self.take_profit:
            lines.append(f"  Take Profit: ${self.take_profit:.2f}")
        if self.stop_loss:
            lines.append(f"  Stop Loss: ${self.stop_loss:.2f}")

        lines.extend([
            f"",
            f"Metrics:",
            f"  Predicted Move: {self.predicted_move_pct:.2f}%",
            f"  Adjusted Edge: {self.adjusted_edge * 100:.3f}%",
            f"  Confidence: {self.confidence:.2f}",
            f"  Risk/Reward: {self.risk_reward_ratio:.2f}:1",
            f"  IC: {self.information_coefficient:.4f}",
            f"  Sharpe: {self.current_sharpe:.2f}",
            f"",
            f"Reasoning:",
        ])

        for reason in self.reasoning:
            lines.append(f"  - {reason}")

        if self.warnings:
            lines.append(f"")
            lines.append(f"Warnings:")
            for warning in self.warnings:
                lines.append(f"  ! {warning}")

        return "\n".join(lines)


class PositionSizer:
    """
    Calculates optimal position sizes based on Sharpe ratio and risk metrics.

    Uses a combination of:
    1. Sharpe-based sizing (higher Sharpe = larger position)
    2. Kelly criterion (optional, conservative fraction)
    3. Volatility adjustment
    4. Drawdown-based reduction
    """

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.historical_returns: List[float] = []
        self.peak_equity: float = 0.0
        self.current_drawdown: float = 0.0

    def update_equity(self, current_equity: float):
        """Update equity tracking for drawdown calculation."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity

    def add_return(self, ret: float):
        """Add a return observation for Sharpe calculation."""
        self.historical_returns.append(ret)
        # Keep last 1000 returns
        if len(self.historical_returns) > 1000:
            self.historical_returns = self.historical_returns[-1000:]

    def calculate_rolling_sharpe(self, window: int = 100) -> float:
        """Calculate rolling Sharpe ratio from recent returns."""
        if len(self.historical_returns) < window:
            if len(self.historical_returns) < 10:
                return 0.0
            returns = np.array(self.historical_returns)
        else:
            returns = np.array(self.historical_returns[-window:])

        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret < 1e-10:
            return 0.0

        # Annualize (assuming hourly returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(8760)
        return float(sharpe)

    def sharpe_based_size(self, current_sharpe: float) -> float:
        """
        Calculate position size based on Sharpe ratio.

        Position = min(Sharpe / Target_Sharpe, Max_Position)
        """
        if current_sharpe <= 0:
            return self.config.min_position

        size = current_sharpe / self.config.target_sharpe
        return float(np.clip(size, self.config.min_position, self.config.max_position))

    def kelly_based_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate position size using Kelly criterion.

        Kelly % = W - [(1-W) / R]
        where W = win rate, R = win/loss ratio

        Uses a conservative fraction of Kelly.
        """
        if avg_loss < 1e-10 or win_rate <= 0:
            return self.config.min_position

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        kelly = win_rate - ((1 - win_rate) / win_loss_ratio) if win_loss_ratio > 0 else 0

        # Use fraction of Kelly for safety
        kelly_size = kelly * self.config.kelly_fraction

        return float(np.clip(kelly_size, self.config.min_position, self.config.max_position))

    def drawdown_adjusted_size(self, base_size: float) -> float:
        """
        Reduce position size based on current drawdown.

        Linear reduction from 100% at 0 drawdown to 50% at max_drawdown_threshold.
        """
        if self.current_drawdown <= 0:
            return base_size

        # Linear reduction
        reduction = min(self.current_drawdown / self.config.max_drawdown_threshold, 1.0)
        adjusted_size = base_size * (1 - 0.5 * reduction)

        return float(np.clip(adjusted_size, self.config.min_position, self.config.max_position))

    def calculate_optimal_size(
        self,
        current_sharpe: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        confidence: float = 1.0
    ) -> Tuple[float, List[str]]:
        """
        Calculate optimal position size using multiple methods.

        Returns:
            Tuple of (position_size, reasoning_list)
        """
        reasoning = []

        # Start with Sharpe-based sizing
        if current_sharpe is None:
            current_sharpe = self.calculate_rolling_sharpe()

        sharpe_size = self.sharpe_based_size(current_sharpe)
        reasoning.append(f"Sharpe-based size: {sharpe_size * 100:.1f}% (Sharpe={current_sharpe:.2f})")

        # Kelly-based sizing (if we have the data)
        if win_rate is not None and avg_win is not None and avg_loss is not None:
            kelly_size = self.kelly_based_size(win_rate, avg_win, avg_loss)
            reasoning.append(f"Kelly-based size: {kelly_size * 100:.1f}%")
            # Average the two approaches
            base_size = (sharpe_size + kelly_size) / 2
            reasoning.append(f"Combined base size: {base_size * 100:.1f}%")
        else:
            base_size = sharpe_size

        # Apply drawdown adjustment
        dd_adjusted = self.drawdown_adjusted_size(base_size)
        if dd_adjusted != base_size:
            reasoning.append(f"Drawdown-adjusted: {dd_adjusted * 100:.1f}% (DD={self.current_drawdown * 100:.1f}%)")

        # Apply confidence scaling
        final_size = dd_adjusted * confidence
        if confidence < 1.0:
            reasoning.append(f"Confidence-adjusted: {final_size * 100:.1f}% (conf={confidence:.2f})")

        final_size = float(np.clip(final_size, self.config.min_position, self.config.max_position))
        reasoning.append(f"Final position size: {final_size * 100:.1f}%")

        return final_size, reasoning


class CapitalAwareExecutor:
    """
    Main execution engine that combines predictions with risk management.

    Provides transparent, capital-aware trade decisions with:
    - Edge calculation and validation
    - Position sizing based on Sharpe
    - Stop loss and take profit levels
    - Full decision transparency and logging
    """

    def __init__(self, config: Optional[ExecutionConfig] = None, verbose: bool = True):
        self.config = config or ExecutionConfig()
        self.verbose = verbose
        self.position_sizer = PositionSizer(self.config)
        self.decision_history: List[TradeDecision] = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        # Ensure log directory exists
        Path(self.config.decision_log_path).parent.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\n[EXECUTOR] CapitalAwareExecutor initialized")
            print(f"[EXECUTOR]   - Min required move: {self.config.min_required_move * 100:.1f}%")
            print(f"[EXECUTOR]   - Target Sharpe: {self.config.target_sharpe}")
            print(f"[EXECUTOR]   - Max position: {self.config.max_position * 100:.1f}%")

    def calculate_edge(
        self,
        predicted_return: float,
        volatility: float,
        horizon: int = 24
    ) -> Tuple[float, float]:
        """
        Calculate and clip edge based on volatility.

        Args:
            predicted_return: Model's predicted return (log return)
            volatility: Current volatility estimate
            horizon: Prediction horizon in periods

        Returns:
            Tuple of (raw_edge, clipped_edge)
        """
        raw_edge = predicted_return

        # Maximum reasonable edge based on volatility
        max_edge = self.config.volatility_mult * volatility * math.sqrt(horizon)

        # Clip edge to reasonable range
        clipped_edge = np.clip(raw_edge, -max_edge, max_edge)

        return float(raw_edge), float(clipped_edge)

    def calculate_levels(
        self,
        price: float,
        edge: float,
        direction: str
    ) -> Tuple[float, float, float]:
        """
        Calculate take profit and stop loss levels.

        Args:
            price: Current price
            edge: Adjusted edge (log return)
            direction: 'BUY' or 'SELL'

        Returns:
            Tuple of (take_profit, stop_loss, risk_reward_ratio)
        """
        abs_edge = abs(edge)

        if direction == 'BUY':
            take_profit = price * math.exp(edge * self.config.confidence_scale)
            stop_loss = price * math.exp(-self.config.stop_mult * abs_edge)
        else:  # SELL
            take_profit = price * math.exp(-abs_edge * self.config.confidence_scale)
            stop_loss = price * math.exp(self.config.stop_mult * abs_edge)

        # Risk/reward ratio
        potential_profit = abs(take_profit - price)
        potential_loss = abs(price - stop_loss)

        if potential_loss > 0:
            risk_reward = potential_profit / potential_loss
        else:
            risk_reward = 0.0

        return float(take_profit), float(stop_loss), float(risk_reward)

    def make_decision(
        self,
        symbol: str,
        current_price: float,
        predicted_return: float,
        volatility: float,
        confidence: float = 1.0,
        ic_score: float = 0.0,
        timestamp: Optional[datetime] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> TradeDecision:
        """
        Make a transparent, capital-aware trading decision.

        This is the main entry point for converting a prediction into
        an actionable trade decision with full reasoning.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            current_price: Current market price
            predicted_return: Model's predicted return (as decimal)
            volatility: Current volatility estimate
            confidence: Model confidence (0-1)
            ic_score: Information Coefficient of recent predictions
            timestamp: Decision timestamp
            additional_context: Extra context for logging

        Returns:
            TradeDecision with full transparency
        """
        timestamp = timestamp or datetime.now()
        reasoning = []
        warnings = []

        # Log the start of decision making
        debug_log("execution", "decision_start", {
            "symbol": symbol,
            "price": current_price,
            "predicted_return": predicted_return,
            "volatility": volatility,
        }, "info")

        # Calculate edge
        raw_edge, adjusted_edge = self.calculate_edge(predicted_return, volatility)
        reasoning.append(f"Raw edge: {raw_edge * 100:.3f}%")
        reasoning.append(f"Volatility-clipped edge: {adjusted_edge * 100:.3f}%")

        # Calculate predicted price and move percentage
        predicted_price = current_price * (1 + predicted_return)
        predicted_move_pct = abs(math.exp(adjusted_edge * self.config.confidence_scale) - 1) * 100

        # Default decision: HOLD
        action = 'HOLD'
        position_size = 0.0
        take_profit = None
        stop_loss = None
        risk_reward = 0.0

        # Check edge threshold
        if abs(adjusted_edge) < self.config.edge_min:
            reasoning.append(f"Edge below threshold ({abs(adjusted_edge) * 100:.3f}% < {self.config.edge_min * 100:.3f}%)")
            reasoning.append("Decision: HOLD - Insufficient edge")
        else:
            # Determine direction
            direction = 'BUY' if adjusted_edge > 0 else 'SELL'
            reasoning.append(f"Direction: {direction} (edge > 0)" if direction == 'BUY' else f"Direction: {direction} (edge < 0)")

            # Check minimum required move
            if predicted_move_pct / 100 < self.config.min_required_move:
                reasoning.append(f"Predicted move {predicted_move_pct:.2f}% below minimum {self.config.min_required_move * 100:.1f}%")
                reasoning.append("Decision: HOLD - Move too small for fees")
            else:
                reasoning.append(f"Predicted move {predicted_move_pct:.2f}% exceeds minimum threshold")

                # Calculate position size
                current_sharpe = self.position_sizer.calculate_rolling_sharpe()

                # Adjust confidence based on IC
                adjusted_confidence = confidence
                if ic_score > 0.10:
                    adjusted_confidence = min(confidence * 1.2, 1.0)
                    reasoning.append(f"IC ({ic_score:.3f}) > 0.10: confidence boosted to {adjusted_confidence:.2f}")
                elif ic_score < 0:
                    adjusted_confidence = confidence * 0.5
                    warnings.append(f"IC negative ({ic_score:.3f}): confidence reduced")

                position_size, sizing_reasons = self.position_sizer.calculate_optimal_size(
                    current_sharpe=current_sharpe,
                    win_rate=self.win_rate if self.total_trades > 10 else None,
                    avg_win=self.avg_win if self.winning_trades > 0 else None,
                    avg_loss=self.avg_loss if self.losing_trades > 0 else None,
                    confidence=adjusted_confidence
                )
                reasoning.extend(sizing_reasons)

                # Calculate take profit and stop loss
                take_profit, stop_loss, risk_reward = self.calculate_levels(current_price, adjusted_edge, direction)
                reasoning.append(f"Take profit: ${take_profit:.2f}")
                reasoning.append(f"Stop loss: ${stop_loss:.2f}")
                reasoning.append(f"Risk/Reward: {risk_reward:.2f}:1")

                # Final decision
                if risk_reward < 1.0:
                    warnings.append(f"Risk/reward ratio below 1:1 ({risk_reward:.2f})")
                    position_size *= 0.5  # Reduce size for poor R/R
                    reasoning.append(f"Position reduced due to low R/R: {position_size * 100:.1f}%")

                if position_size >= self.config.min_position:
                    action = direction
                    reasoning.append(f"Decision: {action} with {position_size * 100:.1f}% position")
                else:
                    reasoning.append(f"Position size too small ({position_size * 100:.1f}% < {self.config.min_position * 100:.1f}%)")
                    reasoning.append("Decision: HOLD - Position size below minimum")
                    position_size = 0.0

        # Create decision object
        decision = TradeDecision(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            position_size=position_size,
            current_price=current_price,
            predicted_price=predicted_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            predicted_return=predicted_return,
            predicted_move_pct=predicted_move_pct,
            raw_edge=raw_edge,
            adjusted_edge=adjusted_edge,
            volatility=volatility,
            confidence=confidence,
            information_coefficient=ic_score,
            current_sharpe=self.position_sizer.calculate_rolling_sharpe(),
            risk_reward_ratio=risk_reward,
            reasoning=reasoning,
            warnings=warnings,
        )

        # Log decision
        if self.config.log_decisions:
            self._log_decision(decision)

        # Store in history
        self.decision_history.append(decision)

        if self.verbose:
            print(f"\n[EXECUTOR] Decision for {symbol}:")
            print(f"[EXECUTOR]   Action: {action}")
            print(f"[EXECUTOR]   Position: {position_size * 100:.1f}%")
            print(f"[EXECUTOR]   Predicted move: {predicted_move_pct:.2f}%")
            if warnings:
                for w in warnings:
                    print(f"[EXECUTOR]   WARNING: {w}")

        debug_log("execution", "decision_complete", decision.to_dict(), "success")

        return decision

    def record_trade_result(self, profit_pct: float):
        """
        Record the result of a completed trade for performance tracking.

        Args:
            profit_pct: Profit/loss as percentage (positive = profit)
        """
        self.total_trades += 1

        if profit_pct > 0:
            self.winning_trades += 1
            self.total_profit += profit_pct
        else:
            self.losing_trades += 1
            self.total_loss += abs(profit_pct)

        # Update position sizer with return
        self.position_sizer.add_return(profit_pct / 100)

    @property
    def win_rate(self) -> float:
        """Calculate win rate from recorded trades."""
        if self.total_trades == 0:
            return 0.5  # Default assumption
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        """Calculate average winning trade percentage."""
        if self.winning_trades == 0:
            return 0.0
        return self.total_profit / self.winning_trades

    @property
    def avg_loss(self) -> float:
        """Calculate average losing trade percentage."""
        if self.losing_trades == 0:
            return 0.01  # Small default to avoid division by zero
        return self.total_loss / self.losing_trades

    def _log_decision(self, decision: TradeDecision):
        """Log decision to file for transparency."""
        try:
            log_path = Path(self.config.decision_log_path)

            # Load existing log or create new
            if log_path.exists():
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'decisions': []}

            # Append new decision
            log_data['decisions'].append(decision.to_dict())

            # Keep only last 1000 decisions
            if len(log_data['decisions']) > 1000:
                log_data['decisions'] = log_data['decisions'][-1000:]

            # Save
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=2)

        except Exception as e:
            if self.verbose:
                print(f"[EXECUTOR] Warning: Failed to log decision: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of executor performance.

        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.total_profit / self.total_loss if self.total_loss > 0 else 0,
            'rolling_sharpe': self.position_sizer.calculate_rolling_sharpe(),
            'current_drawdown': self.position_sizer.current_drawdown,
            'total_decisions': len(self.decision_history),
        }

    def print_performance(self):
        """Print formatted performance summary."""
        perf = self.get_performance_summary()

        print(f"\n{'='*60}")
        print(f"EXECUTOR PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total Trades: {perf['total_trades']}")
        print(f"Win Rate: {perf['win_rate'] * 100:.1f}%")
        print(f"Average Win: {perf['avg_win']:.2f}%")
        print(f"Average Loss: {perf['avg_loss']:.2f}%")
        print(f"Profit Factor: {perf['profit_factor']:.2f}")
        print(f"Rolling Sharpe: {perf['rolling_sharpe']:.2f}")
        print(f"Current Drawdown: {perf['current_drawdown'] * 100:.1f}%")
        print(f"Total Decisions: {perf['total_decisions']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    # Test the executor
    print("Testing CapitalAwareExecutor")
    print("=" * 60)

    executor = CapitalAwareExecutor()

    # Simulate some decisions
    test_cases = [
        {"symbol": "BTCUSDT", "price": 50000, "predicted_return": 0.03, "vol": 0.02, "conf": 0.8},
        {"symbol": "BTCUSDT", "price": 50000, "predicted_return": 0.005, "vol": 0.02, "conf": 0.6},
        {"symbol": "ETHUSDT", "price": 3000, "predicted_return": -0.025, "vol": 0.025, "conf": 0.7},
        {"symbol": "BTCUSDT", "price": 51000, "predicted_return": 0.001, "vol": 0.02, "conf": 0.9},
    ]

    for tc in test_cases:
        decision = executor.make_decision(
            symbol=tc["symbol"],
            current_price=tc["price"],
            predicted_return=tc["predicted_return"],
            volatility=tc["vol"],
            confidence=tc["conf"],
        )
        print(decision.summary())
        print()

    executor.print_performance()
