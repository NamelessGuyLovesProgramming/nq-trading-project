import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.plugins.base_plugin import BaseStrategyPlugin, StrategyParameter


class CustomMACDStrategy(BaseStrategyPlugin):
    """
    Benutzerdefinierte MACD-Strategie mit anpassbaren Parametern.

    Diese Strategie kombiniert MACD mit Volumen-Bestätigung und
    optionalem ATR-basierten Risikomanagement.
    """

    def get_name(self) -> str:
        return "Custom MACD Volume Strategy"

    def get_description(self) -> str:
        return """
        Eine Strategie, die auf MACD-Signalen basiert, mit Volumen-Bestätigung 
        und optionalem ATR-basierten Risikomanagement.

        Kaufsignal:
        - MACD überquert Signal-Linie von unten nach oben
        - Volumen ist über dem gleitenden Durchschnitt des Volumens

        Verkaufssignal:
        - MACD überquert Signal-Linie von oben nach unten
        - Volumen ist über dem gleitenden Durchschnitt des Volumens
        """

    def get_parameters(self) -> List[StrategyParameter]:
        return [
            StrategyParameter(
                name="fast_period",
                type="int",
                default=12,
                description="Periode für schnellen EMA im MACD",
                min_value=2,
                max_value=50
            ),
            StrategyParameter(
                name="slow_period",
                type="int",
                default=26,
                description="Periode für langsamen EMA im MACD",
                min_value=5,
                max_value=100
            ),
            StrategyParameter(
                name="signal_period",
                type="int",
                default=9,
                description="Periode für MACD Signal-Linie",
                min_value=2,
                max_value=50
            ),
            StrategyParameter(
                name="volume_threshold",
                type="float",
                default=1.0,
                description="Volumen-Schwellenwert (als Vielfaches des Durchschnitts)",
                min_value=0.5,
                max_value=3.0
            ),
            StrategyParameter(
                name="volume_period",
                type="int",
                default=20,
                description="Periode für gleitenden Durchschnitt des Volumens",
                min_value=5,
                max_value=100
            ),
            StrategyParameter(
                name="use_atr_filter",
                type="bool",
                default=True,
                description="ATR-Filter für Volatilität verwenden"
            ),
            StrategyParameter(
                name="atr_period",
                type="int",
                default=14,
                description="Periode für ATR-Berechnung",
                min_value=5,
                max_value=50
            ),
            StrategyParameter(
                name="atr_multiplier",
                type="float",
                default=1.5,
                description="ATR-Multiplikator für Schwellenwert",
                min_value=0.5,
                max_value=5.0
            )
        ]

    def generate_signals(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Generiert Handelssignale basierend auf MACD und Volumen.

        Args:
            data: DataFrame mit OHLCV-Daten und Indikatoren
            params: Dictionary mit Parametern

        Returns:
            DataFrame mit 'Signal'-Spalte
        """
        # Kopie des DataFrames erstellen
        df = data.copy()

        # Parameter extrahieren
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)
        signal_period = params.get("signal_period", 9)
        volume_threshold = params.get("volume_threshold", 1.0)
        volume_period = params.get("volume_period", 20)
        use_atr_filter = params.get("use_atr_filter", True)
        atr_period = params.get("atr_period", 14)
        atr_multiplier = params.get("atr_multiplier", 1.5)

        # MACD berechnen (falls nicht bereits im DataFrame vorhanden)
        if 'MACD' not in df.columns or 'MACD_Signal' not in df.columns:
            # Berechne EMA für MACD
            ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()

            # MACD-Linie
            df['MACD'] = ema_fast - ema_slow

            # Signal-Linie
            df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()

            # MACD-Histogramm
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Volumen-Verhältnis berechnen
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=volume_period).mean()

        # ATR berechnen, falls erforderlich
        if use_atr_filter and 'ATR' not in df.columns:
            # True Range berechnen
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # ATR berechnen
            df['ATR'] = true_range.rolling(window=atr_period).mean()

        # Signal-Logik
        df['Signal'] = 0  # Initialisiere mit Null (kein Signal)

        # MACD-Kreuzungen identifizieren
        df['MACD_Cross_Up'] = (df['MACD'].shift(1) < df['MACD_Signal'].shift(1)) & (df['MACD'] > df['MACD_Signal'])
        df['MACD_Cross_Down'] = (df['MACD'].shift(1) > df['MACD_Signal'].shift(1)) & (df['MACD'] < df['MACD_Signal'])

        # Hohe Volumen-Bedingung
        high_volume = df['Volume_Ratio'] > volume_threshold

        # ATR-Filter für hohe Volatilität, falls aktiviert
        if use_atr_filter:
            # Berechne ATR-zu-Preis-Verhältnis
            df['ATR_Ratio'] = df['ATR'] / df['Close']

            # Berechne gleitenden Durchschnitt des ATR-Verhältnisses
            df['ATR_Ratio_Avg'] = df['ATR_Ratio'].rolling(window=volume_period).mean()

            # Signale durch ATR-Filter filtern
            atr_filter = df['ATR_Ratio'] < (df['ATR_Ratio_Avg'] * atr_multiplier)

            # Generiere Signale unter Berücksichtigung aller Bedingungen
            df.loc[(df['MACD_Cross_Up']) & high_volume & atr_filter, 'Signal'] = 1  # Kaufsignal
            df.loc[(df['MACD_Cross_Down']) & high_volume & atr_filter, 'Signal'] = -1  # Verkaufssignal
        else:
            # Generiere Signale ohne ATR-Filter
            df.loc[(df['MACD_Cross_Up']) & high_volume, 'Signal'] = 1  # Kaufsignal
            df.loc[(df['MACD_Cross_Down']) & high_volume, 'Signal'] = -1  # Verkaufssignal

        return df