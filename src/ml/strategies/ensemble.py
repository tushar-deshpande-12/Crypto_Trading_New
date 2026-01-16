"""
Ensemble Strategy - Combines multiple strategies with weighted voting

This strategy aggregates signals from multiple base strategies
to produce more robust trading decisions.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import register_strategy, get_strategy


@register_strategy
class EnsembleStrategy(BaseStrategy):
    """
    Ensemble Strategy with Weighted Voting

    Combines signals from multiple strategies using configurable voting methods:
    - 'majority': Action with most votes wins
    - 'weighted': Weighted by strategy confidence and weight
    - 'unanimous': All strategies must agree for action

    Example:
        ensemble = EnsembleStrategy(config=StrategyConfig(
            name='ensemble',
            params={
                'strategies': ['rsi', 'macd', 'bollinger'],
                'weights': [0.4, 0.3, 0.3],
                'voting_method': 'weighted'
            }
        ))
    """

    @property
    def name(self) -> str:
        return "ensemble"

    @property
    def description(self) -> str:
        return "Ensemble Strategy - Combines multiple strategies with weighted voting"

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        params = self.config.params

        # Get strategy names and weights
        strategy_names = params.get('strategies', ['rsi', 'macd', 'bollinger'])
        weights = params.get('weights', None)

        if weights is None:
            # Equal weights by default
            weights = [1.0 / len(strategy_names)] * len(strategy_names)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]

        self.voting_method = params.get('voting_method', 'weighted')
        self.min_agreement = params.get('min_agreement', 0.5)  # Minimum vote share to act

        # Initialize sub-strategies
        self.strategies: List[Tuple[BaseStrategy, float]] = []
        for name, weight in zip(strategy_names, weights):
            try:
                strategy = get_strategy(name)
                self.strategies.append((strategy, weight))
            except KeyError:
                print(f"Warning: Strategy '{name}' not found, skipping")

    def get_default_params(self) -> Dict[str, Any]:
        return {
            'strategies': ['rsi', 'macd', 'bollinger'],
            'weights': None,  # Equal weights
            'voting_method': 'weighted',  # 'majority', 'weighted', 'unanimous'
            'min_agreement': 0.5  # Minimum vote share to act
        }

    def get_required_features(self) -> List[str]:
        """Collect required features from all sub-strategies"""
        features = set()
        for strategy, _ in self.strategies:
            features.update(strategy.get_required_features())
        return list(features)

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate ensemble signals using voting"""
        # First, generate signals from all sub-strategies
        all_signals: List[List[TradeSignal]] = []
        for strategy, _ in self.strategies:
            try:
                signals = strategy.generate_signals(data)
                all_signals.append(signals)
            except Exception as e:
                print(f"Warning: Strategy {strategy.name} failed: {e}")
                # Create HOLD signals as fallback
                hold_signals = [
                    TradeSignal(action='HOLD', confidence=0.0, price=row.get('close', 0))
                    for _, row in data.iterrows()
                ]
                all_signals.append(hold_signals)

        # Combine signals
        combined_signals = []
        for idx in range(len(data)):
            row = data.iloc[idx]
            price = row.get('close', 0)
            timestamp = row.get('datetime', None)

            # Collect votes
            votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
            indicator_values = {}
            strategy_signals = {}

            for (strategy, weight), signals in zip(self.strategies, all_signals):
                if idx < len(signals):
                    signal = signals[idx]

                    if self.voting_method == 'weighted':
                        # Weight by strategy weight AND signal confidence
                        votes[signal.action] += weight * signal.confidence
                    elif self.voting_method == 'majority':
                        # Simple vote
                        votes[signal.action] += weight
                    elif self.voting_method == 'unanimous':
                        # All must agree
                        votes[signal.action] += weight

                    # Store for metadata
                    strategy_signals[strategy.name] = {
                        'action': signal.action,
                        'confidence': signal.confidence
                    }
                    indicator_values.update(signal.indicator_values)

            # Determine winner
            if self.voting_method == 'unanimous':
                # All strategies must signal same action
                actions = [s['action'] for s in strategy_signals.values()]
                if len(set(actions)) == 1 and actions[0] != 'HOLD':
                    action = actions[0]
                    confidence = sum(s['confidence'] for s in strategy_signals.values()) / len(strategy_signals)
                else:
                    action = 'HOLD'
                    confidence = 0.0
            else:
                # Majority or weighted voting
                action = max(votes, key=votes.get)
                vote_share = votes[action]

                # Apply minimum agreement threshold
                if vote_share < self.min_agreement:
                    action = 'HOLD'
                    confidence = 0.0
                else:
                    confidence = min(vote_share, 1.0)

            combined_signals.append(TradeSignal(
                action=action,
                confidence=confidence,
                price=price,
                timestamp=timestamp,
                indicator_values=indicator_values,
                metadata={
                    'votes': votes,
                    'strategy_signals': strategy_signals,
                    'voting_method': self.voting_method
                }
            ))

        return combined_signals


# Convenience function to create ensemble with specific strategies
def create_ensemble(strategy_names: List[str],
                   weights: Optional[List[float]] = None,
                   voting_method: str = 'weighted') -> EnsembleStrategy:
    """
    Create an ensemble strategy with specified components.

    Args:
        strategy_names: List of strategy names to include
        weights: Optional weights for each strategy (default: equal)
        voting_method: 'majority', 'weighted', or 'unanimous'

    Returns:
        Configured EnsembleStrategy instance
    """
    config = StrategyConfig(
        name='ensemble',
        params={
            'strategies': strategy_names,
            'weights': weights,
            'voting_method': voting_method
        }
    )
    return EnsembleStrategy(config=config)
