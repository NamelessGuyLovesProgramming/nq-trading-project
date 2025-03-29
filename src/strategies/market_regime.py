# src/strategies/market_regime.py
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.utils.helpers import prepare_dataframe


class MarketRegimeStrategy(BaseStrategy):
    def __init__(self):
        """
        Strategie, die sich an verschiedene Marktphasen anpasst.
        """
        super().__init__(name="Market_Regime_Strategy")

    def detect_market_regime(self, data):
        """
        Erkennt die aktuelle Marktphase.

        Returns:
        --------
        str
            'trend_up', 'trend_down', 'range_bound', 'high_volatility'
        """
        df = prepare_dataframe(data.copy())

        # Berechne aktuelle Volatilität (z.B. ATR/Close)
        if 'ATR' not in df.columns:
            # Berechne ATR
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df['Close'].shift()).abs()
            low_close = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()

        vol_ratio = df['ATR'] / df['Close']
        avg_vol_ratio = vol_ratio.rolling(30).mean().iloc[-1]
        current_vol_ratio = vol_ratio.iloc[-1]

        # Trend-Stärke (z.B. mittels ADX oder einfacher Trendbestimmung)
        trend_strength = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]

        # Bestimme Marktphase
        if current_vol_ratio > avg_vol_ratio * 1.5:
            return 'high_volatility'
        elif trend_strength > 0.05:  # 5% Aufwärtstrend
            return 'trend_up'
        elif trend_strength < -0.05:  # 5% Abwärtstrend
            return 'trend_down'
        else:
            return 'range_bound'

    def generate_signals(self, data):
        # Daten für die Verarbeitung vorbereiten
        df = prepare_dataframe(data.copy())
        df['Signal'] = 0
        df['Market_Regime'] = None

        # Verwende ein rollierendes Fenster für die Marktphase
        lookback = 20
        for i in range(lookback, len(df)):
            window_data = df.iloc[i - lookback:i + 1]
            regime = self.detect_market_regime(window_data)
            df.iloc[i, df.columns.get_loc('Market_Regime')] = regime

            # Wähle Strategie basierend auf der Marktphase
            if regime == 'trend_up':
                # Trendfolge-Strategie für Aufwärtstrend
                if (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i]) and (df['Close'].iloc[i] > df['SMA_50'].iloc[i]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 1

            elif regime == 'trend_down':
                # Trendfolge-Strategie für Abwärtstrend
                if (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i]) and (df['Close'].iloc[i] < df['SMA_50'].iloc[i]):
                    df.iloc[i, df.columns.get_loc('Signal')] = -1

            elif regime == 'range_bound':
                # Mean-Reversion für Seitwärtsmärkte
                if (df['RSI'].iloc[i] < 30) and (df['Close'].iloc[i] < df['BB_Lower'].iloc[i]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 1
                elif (df['RSI'].iloc[i] > 70) and (df['Close'].iloc[i] > df['BB_Upper'].iloc[i]):
                    df.iloc[i, df.columns.get_loc('Signal')] = -1

            elif regime == 'high_volatility':
                # Vorsichtige Strategie für hohe Volatilität
                # In volatilen Märkten weniger handeln oder kurzfristigere Ziele setzen
                pass

        return df