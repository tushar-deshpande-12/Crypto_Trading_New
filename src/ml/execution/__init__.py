"""
Execution module for capital-aware trading decisions.
"""

from .capital_aware import (
    CapitalAwareExecutor,
    ExecutionConfig,
    TradeDecision,
    PositionSizer,
)

__all__ = [
    'CapitalAwareExecutor',
    'ExecutionConfig',
    'TradeDecision',
    'PositionSizer',
]
