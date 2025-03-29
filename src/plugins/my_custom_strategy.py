# src/plugins/my_custom_strategy.py
from src.plugins.strategy_plugin import StrategyPlugin


class MyCustomStrategy(StrategyPlugin):
    def get_name(self):
        return "My Custom Strategy"

    def get_parameters(self):
        return [
            {"name": "lookback", "type": "int", "default": 20, "description": "Lookback period"},
            {"name": "threshold", "type": "float", "default": 0.5, "description": "Signal threshold"}
        ]

    def generate_signals(self, data, params):
        # Implementiere deine Strategie-Logik hier
        # ...
        return signals