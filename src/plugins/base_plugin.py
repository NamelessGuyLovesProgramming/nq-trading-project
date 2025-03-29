import abc
import inspect
import os
import importlib.util
import pandas as pd
from typing import Dict, Any, List, Union, Tuple, Optional


class StrategyParameter:
    """
    Repräsentiert einen Parameter für eine Trading-Strategie.
    """

    def __init__(
            self,
            name: str,
            type: str,
            default: Any,
            description: str,
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,
            choices: Optional[List[Any]] = None
    ):
        """
        Initialisiert einen Strategie-Parameter.

        Args:
            name: Name des Parameters
            type: Datentyp ('int', 'float', 'bool', 'str', 'select')
            default: Standardwert
            description: Beschreibung für UI
            min_value: Minimalwert (für numerische Parameter)
            max_value: Maximalwert (für numerische Parameter)
            choices: Auswahlmöglichkeiten (für type='select')
        """
        self.name = name
        self.type = type
        self.default = default
        self.description = description
        self.min_value = min_value
        self.max_value = max_value
        self.choices = choices

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Parameter in ein Dictionary."""
        result = {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "description": self.description
        }

        if self.min_value is not None:
            result["min_value"] = self.min_value

        if self.max_value is not None:
            result["max_value"] = self.max_value

        if self.choices is not None:
            result["choices"] = self.choices

        return result


class BaseStrategyPlugin(abc.ABC):
    """
    Basis-Klasse für Strategie-Plugins im NQ-Trading-Tool.

    Alle benutzerdefinierten Strategien müssen von dieser Klasse erben
    und die abstrakten Methoden implementieren.
    """

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Gibt den Namen der Strategie zurück.
        """
        pass

    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Gibt eine Beschreibung der Strategie zurück.
        """
        pass

    @abc.abstractmethod
    def get_parameters(self) -> List[StrategyParameter]:
        """
        Gibt eine Liste von Parametern zurück, die die UI anzeigen soll.
        """
        pass

    @abc.abstractmethod
    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf den Daten und Parametern.

        Args:
            data: DataFrame mit OHLCV-Daten und Indikatoren
            params: Dictionary mit Parametern

        Returns:
            DataFrame mit zusätzlicher 'Signal'-Spalte
        """
        pass

    def calculate_returns(
            self,
            data: pd.DataFrame,
            signals: pd.DataFrame,
            initial_capital: float = 10000.0,
            commission: float = 0.001
    ) -> Dict[str, Any]:
        """
        Berechnet die Renditen der Strategie.

        Diese Methode kann überschrieben werden, wenn eine spezielle Berechnung
        erforderlich ist, sonst wird die Standardimplementierung verwendet.

        Args:
            data: DataFrame mit OHLCV-Daten
            signals: DataFrame mit Signalen
            initial_capital: Anfangskapital
            commission: Kommissionssatz

        Returns:
            Dictionary mit Backtest-Ergebnissen
        """
        # Standardimplementierung verwenden
        from src.strategies.base import BaseStrategy

        # Erstelle eine temporäre Klasse, die von BaseStrategy erbt
        class TempStrategy(BaseStrategy):
            def __init__(self, name):
                super().__init__(name=name)

            def generate_signals(self, df):
                return signals

        # Instanziiere die Strategie und berechne Renditen
        temp_strategy = TempStrategy(self.get_name())
        return temp_strategy.calculate_returns(data, signals, initial_capital, commission)


class IndicatorPlugin(abc.ABC):
    """
    Basis-Klasse für benutzerdefinierte technische Indikatoren.
    """

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Gibt den Namen des Indikators zurück.
        """
        pass

    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Gibt eine Beschreibung des Indikators zurück.
        """
        pass

    @abc.abstractmethod
    def get_parameters(self) -> List[StrategyParameter]:
        """
        Gibt eine Liste von Parametern zurück, die die UI anzeigen soll.
        """
        pass

    @abc.abstractmethod
    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Berechnet den Indikator basierend auf den Daten und Parametern.

        Args:
            data: DataFrame mit OHLCV-Daten
            params: Dictionary mit Parametern

        Returns:
            Dictionary mit Spaltennamen und berechneten Serien
        """
        pass


class PluginManager:
    """
    Manager für Plugins (Strategien und Indikatoren).
    """

    def __init__(self, plugins_dir: str = "src/plugins"):
        """
        Initialisiert den PluginManager.

        Args:
            plugins_dir: Verzeichnis mit Plugins
        """
        self.plugins_dir = plugins_dir
        self.strategy_plugins = {}
        self.indicator_plugins = {}

        # Lade Plugins beim Start
        self.load_plugins()

    def load_plugins(self):
        """
        Lädt alle verfügbaren Plugins aus dem Plugins-Verzeichnis.
        """
        if not os.path.exists(self.plugins_dir):
            print(f"Plugins-Verzeichnis nicht gefunden: {self.plugins_dir}")
            return

        # Durchsuche Unterverzeichnisse
        for root, dirs, files in os.walk(self.plugins_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__") and file != "base_plugin.py":
                    try:
                        # Lade Modul
                        module_path = os.path.join(root, file)
                        module_name = os.path.splitext(file)[0]
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Suche nach Plugin-Klassen
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and not inspect.isabstract(obj):
                                # Prüfe, ob es sich um ein Strategy-Plugin handelt
                                if issubclass(obj, BaseStrategyPlugin) and obj != BaseStrategyPlugin:
                                    plugin = obj()
                                    self.strategy_plugins[plugin.get_name()] = plugin
                                    print(f"Strategie-Plugin geladen: {plugin.get_name()}")

                                # Prüfe, ob es sich um ein Indicator-Plugin handelt
                                elif issubclass(obj, IndicatorPlugin) and obj != IndicatorPlugin:
                                    plugin = obj()
                                    self.indicator_plugins[plugin.get_name()] = plugin
                                    print(f"Indikator-Plugin geladen: {plugin.get_name()}")

                    except Exception as e:
                        print(f"Fehler beim Laden des Plugins {file}: {e}")

    def get_strategy_plugin(self, name: str) -> BaseStrategyPlugin:
        """
        Gibt ein Strategie-Plugin anhand des Namens zurück.

        Args:
            name: Name des Plugins

        Returns:
            Strategie-Plugin

        Raises:
            KeyError: Wenn das Plugin nicht gefunden wurde
        """
        if name not in self.strategy_plugins:
            raise KeyError(f"Strategie-Plugin nicht gefunden: {name}")

        return self.strategy_plugins[name]

    def get_indicator_plugin(self, name: str) -> IndicatorPlugin:
        """
        Gibt ein Indikator-Plugin anhand des Namens zurück.

        Args:
            name: Name des Plugins

        Returns:
            Indikator-Plugin

        Raises:
            KeyError: Wenn das Plugin nicht gefunden wurde
        """
        if name not in self.indicator_plugins:
            raise KeyError(f"Indikator-Plugin nicht gefunden: {name}")

        return self.indicator_plugins[name]

    def get_all_strategy_plugins(self) -> Dict[str, BaseStrategyPlugin]:
        """
        Gibt alle verfügbaren Strategie-Plugins zurück.

        Returns:
            Dictionary mit Plugin-Namen und Instanzen
        """
        return self.strategy_plugins

    def get_all_indicator_plugins(self) -> Dict[str, IndicatorPlugin]:
        """
        Gibt alle verfügbaren Indikator-Plugins zurück.

        Returns:
            Dictionary mit Plugin-Namen und Instanzen
        """
        return self.indicator_plugins