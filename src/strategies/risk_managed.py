# src/strategies/risk_managed.py
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.utils.helpers import prepare_dataframe


class RiskManagedStrategy(BaseStrategy):
    def __init__(self, base_strategy, max_drawdown=-0.05, volatility_filter=True,
                 risk_per_trade=0.01, position_size_method='fixed', atr_risk_multiplier=1.5):
        """
        Strategie-Wrapper mit Risikomanagement.

        Parameters:
        -----------
        base_strategy : BaseStrategy
            Basis-Strategie für Signale
        max_drawdown : float
            Maximaler Drawdown, bevor die Positionsgröße reduziert wird
        volatility_filter : bool
            Reduziert Positionsgröße bei hoher Volatilität
        risk_per_trade : float
            Prozentsatz des Kapitals, das pro Trade riskiert wird (z.B. 0.01 = 1%)
        position_size_method : str
            Methode zur Berechnung der Positionsgröße ('fixed', 'percent', 'atr')
        atr_risk_multiplier : float
            Multiplikator für ATR bei ATR-basierter Positionsgrößenberechnung
        """
        super().__init__(name=f"Risk_Managed_{base_strategy.name}")
        self.base_strategy = base_strategy
        self.max_drawdown = max_drawdown
        self.volatility_filter = volatility_filter
        self.risk_per_trade = risk_per_trade
        self.position_size_method = position_size_method
        self.atr_risk_multiplier = atr_risk_multiplier

    def calculate_position_size(self, df, i, capital=None):
        """
        Berechnet die optimale Positionsgröße basierend auf Risikometriken.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame mit OHLCV- und Indikatordaten
        i : int
            Index der aktuellen Zeile
        capital : float
            Aktuelles Kapital (falls verfügbar)

        Returns:
        --------
        float
            Positionsgröße (0.0 bis 1.0, wobei 1.0 = volle Größe)
        """
        # Standardgröße
        position_size = 1.0

        # 1. Drawdown-basierte Anpassung
        if 'Drawdown' in df.columns:
            current_drawdown = df['Drawdown'].iloc[i]
            if current_drawdown < self.max_drawdown:
                # Reduziere Positionsgröße proportional zum Drawdown
                drawdown_factor = max(0.2, 1 + (current_drawdown / self.max_drawdown))
                position_size *= drawdown_factor

        # 2. Volatilitäts-basierte Anpassung
        if self.volatility_filter and 'ATR' in df.columns:
            # Berechne relative Volatilität
            atr_ratio = df['ATR'].iloc[i] / df['Close'].iloc[i]
            atr_avg = df['ATR'].iloc[max(0, i - 20):i + 1].mean() / df['Close'].iloc[max(0, i - 20):i + 1].mean()

            if atr_ratio > atr_avg * 1.5:
                # Hohe Volatilität - reduziere Position
                vol_factor = atr_avg / atr_ratio
                position_size *= vol_factor

        # 3. Positionsgrößenanpassung basierend auf der gewählten Methode
        if self.position_size_method == 'fixed':
            # Feste Positionsgröße (bereits durch position_size bestimmt)
            pass

        elif self.position_size_method == 'percent' and capital is not None:
            # Position basierend auf Prozentsatz des Kapitals
            position_size *= self.risk_per_trade

        elif self.position_size_method == 'atr' and 'ATR' in df.columns and capital is not None:
            # ATR-basierte Positionsgröße
            # Berechne Stop-Loss-Abstand basierend auf ATR
            atr_value = df['ATR'].iloc[i]
            stop_loss_distance = atr_value * self.atr_risk_multiplier

            # Berechne maximalen Verlust pro Kontrakt/Aktie
            price = df['Close'].iloc[i]
            max_loss_per_unit = stop_loss_distance

            # Maximaler Verlust in Dollar, den wir bereit sind zu akzeptieren
            max_dollar_risk = capital * self.risk_per_trade

            # Berechne Anzahl der Einheiten
            if max_loss_per_unit > 0:
                units = max_dollar_risk / max_loss_per_unit
                # Normalisiere auf Position Size
                normalized_size = min(1.0, units / (capital / price))
                position_size *= normalized_size

        return max(0.0, min(1.0, position_size))  # Begrenze auf 0-100%

    def generate_signals(self, data):
        """
        Generiert Handelssignale mit Risikomanagement.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame mit OHLCV- und Indikatordaten

        Returns:
        --------
        pd.DataFrame
            DataFrame mit Signalen und Positionsgrößen
        """
        # Daten für die Verarbeitung vorbereiten
        df = prepare_dataframe(data.copy())

        # Basissignale von der Strategie bekommen
        base_signals = self.base_strategy.generate_signals(df)

        # Stelle sicher, dass base_signals die richtige Struktur hat
        base_signals = prepare_dataframe(base_signals)

        # Berechne Drawdown für die Risikoanpassung
        if 'Drawdown' not in base_signals.columns:
            base_signals['Returns'] = base_signals['Close'].pct_change()
            base_signals['Cumulative_Returns'] = (1 + base_signals['Returns']).cumprod()
            base_signals['Peak'] = base_signals['Cumulative_Returns'].cummax()
            base_signals['Drawdown'] = (base_signals['Cumulative_Returns'] - base_signals['Peak']) / base_signals[
                'Peak']

        # Füge Positionsgröße hinzu
        base_signals['Position_Size'] = 1.0  # Standardwert

        # Simuliere Kapitalentwicklung für genauere Positionsgrößenberechnung
        initial_capital = 50000.0  # Standard-Anfangskapital als float, nicht int

        # Stelle sicher, dass Simulated_Capital immer als float behandelt wird
        base_signals['Simulated_Capital'] = initial_capital

        # Berechne Positionsgrößen für jeden Zeitpunkt
        for i in range(len(base_signals)):
            if i > 0:
                # Aktualisiere simuliertes Kapital basierend auf vorherigen Signalen
                prev_signal = base_signals['Signal'].iloc[i - 1]
                prev_pos_size = base_signals['Position_Size'].iloc[i - 1]
                returns = base_signals['Returns'].iloc[i]

                # Berechne Kapitaländerung
                capital_change = prev_signal * prev_pos_size * returns * base_signals['Simulated_Capital'].iloc[i - 1]

                # Speichere das neue Kapital als float
                new_capital = base_signals['Simulated_Capital'].iloc[i - 1] + capital_change

                # Sichere Zuweisung des float-Werts zur Simulated_Capital-Spalte
                base_signals.loc[base_signals.index[i], 'Simulated_Capital'] = new_capital

            # Berechne Positionsgröße mit aktuellem Kapital
            current_capital = base_signals['Simulated_Capital'].iloc[i]
            position_size = self.calculate_position_size(base_signals, i, current_capital)

            # Sichere Zuweisung des float-Werts zur Position_Size-Spalte
            base_signals.loc[base_signals.index[i], 'Position_Size'] = position_size

        # Passe die Signalstärke basierend auf der Positionsgröße an
        # (für das Backtesting kann dies als Anteil des investierten Kapitals interpretiert werden)
        base_signals['Signal'] = base_signals['Signal'] * base_signals['Position_Size']

        return base_signals