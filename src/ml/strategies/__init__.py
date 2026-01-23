# Trading Strategies Module
# Provides pluggable trading strategies for backtesting

from .base import BaseStrategy, TradeSignal, StrategyConfig
from .registry import StrategyRegistry, get_strategy, list_strategies, register_strategy

# Import strategies to trigger registration
from .rsi_strategy import RSIStrategy
from .macd_strategy import MACDStrategy
from .bollinger_strategy import BollingerStrategy
from .ma_crossover_strategy import MACrossoverStrategy
from .stochastic_strategy import StochasticStrategy
from .prediction_strategy import PredictionStrategy
from .ensemble import EnsembleStrategy, create_ensemble

# Combo strategies (famous indicator combinations)
from .combo_strategies import (
    RSIMACDComboStrategy,
    BollingerRSIComboStrategy,
    MACDMAComboStrategy,
    StochRSIComboStrategy,
    TripleEMAStrategy,
    ADXMACDStrategy,
)

__all__ = [
    # Base classes
    'BaseStrategy',
    'TradeSignal',
    'StrategyConfig',

    # Registry
    'StrategyRegistry',
    'get_strategy',
    'list_strategies',
    'register_strategy',

    # Basic Strategies
    'RSIStrategy',
    'MACDStrategy',
    'BollingerStrategy',
    'MACrossoverStrategy',
    'StochasticStrategy',
    'PredictionStrategy',
    'EnsembleStrategy',
    'create_ensemble',

    # Combo Strategies
    'RSIMACDComboStrategy',
    'BollingerRSIComboStrategy',
    'MACDMAComboStrategy',
    'StochRSIComboStrategy',
    'TripleEMAStrategy',
    'ADXMACDStrategy',
]
