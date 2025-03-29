# src/plugins/strategy_plugin.py
class StrategyPlugin:
    """Basis-Klasse für Strategie-Plugins"""

    def get_name(self):
        """Gibt den Namen der Strategie zurück"""
        raise NotImplementedError

    def get_parameters(self):
        """Gibt eine Liste von Parametern zurück, die die UI anzeigen soll"""
        raise NotImplementedError

    def generate_signals(self, data, params):
        """Generiert Handelssignale basierend auf den Daten und Parametern"""
        raise NotImplementedError