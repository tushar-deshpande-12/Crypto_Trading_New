"""
Strategy Registry - Dynamic strategy loading and management

This module provides a registry for trading strategies, allowing
dynamic loading and instantiation of strategies by name.
"""

from typing import Dict, List, Type, Optional
from .base import BaseStrategy, StrategyConfig


class StrategyRegistry:
    """
    Registry for trading strategies.

    Provides centralized management of strategy classes, allowing
    strategies to be registered and retrieved by name.
    """

    _strategies: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
        """
        Register a strategy class.

        Can be used as a decorator:
            @StrategyRegistry.register
            class MyStrategy(BaseStrategy):
                ...

        Args:
            strategy_class: The strategy class to register

        Returns:
            The registered class (for decorator use)
        """
        # Create a temporary instance to get the name property value
        try:
            temp_instance = strategy_class(config=StrategyConfig(name="temp"))
            name = temp_instance.name
        except Exception:
            # Fallback to class name if instantiation fails
            name = strategy_class.__name__.lower().replace('strategy', '')

        cls._strategies[name] = strategy_class
        return strategy_class

    @classmethod
    def get(cls, name: str, config: Optional[StrategyConfig] = None) -> BaseStrategy:
        """
        Get a strategy instance by name.

        Args:
            name: Strategy name
            config: Optional configuration

        Returns:
            Strategy instance

        Raises:
            KeyError: If strategy not found
        """
        if name not in cls._strategies:
            raise KeyError(f"Strategy '{name}' not found. Available: {list(cls._strategies.keys())}")

        strategy_class = cls._strategies[name]
        return strategy_class(config=config)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """Get list of registered strategy names"""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls) -> Dict[str, str]:
        """Get dict of strategy names and descriptions"""
        info = {}
        for name, strategy_class in cls._strategies.items():
            try:
                temp = object.__new__(strategy_class)
                temp.__init__(config=StrategyConfig(name=name))
                info[name] = temp.description
            except Exception:
                info[name] = "No description available"
        return info

    @classmethod
    def clear(cls):
        """Clear all registered strategies (for testing)"""
        cls._strategies.clear()


# Convenience functions
def get_strategy(name: str, config: Optional[StrategyConfig] = None) -> BaseStrategy:
    """Get a strategy instance by name"""
    return StrategyRegistry.get(name, config)


def list_strategies() -> List[str]:
    """Get list of available strategy names"""
    return StrategyRegistry.list_strategies()


def register_strategy(strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """Register a strategy class (decorator)"""
    return StrategyRegistry.register(strategy_class)
