# src/strategies/combined.py
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.utils.helpers import prepare_dataframe


class CombinedStrategy(BaseStrategy):
    def __init__(self, weights={'trend': 0.4, 'momentum': 0.3, 'volatility': 0.3}, threshold=0.2):
        """
        Kombiniert verschiedene Signalquellen mit Gewichtung.

        Parameters:
        -----------
        weights : dict
            Gewichtungen für verschiedene Signaltypen
        threshold : float
            Schwellenwert für das kombinierte Signal
        """
        super().__init__(name="Combined_Strategy")
        self.weights = weights
        self.threshold = threshold

    def generate_signals(self, data):
        # Daten für die Verarbeitung vorbereiten
        df = prepare_dataframe(data.copy())

        # 1. Trend-Signale
        df['Trend_Signal'] = 0
        df.loc[(df['Close'] > df['SMA_50']) & (df['SMA_20'] > df['SMA_50']), 'Trend_Signal'] = 1
        df.loc[(df['Close'] < df['SMA_50']) & (df['SMA_20'] < df['SMA_50']), 'Trend_Signal'] = -1

        # 2. Momentum-Signale
        df['Momentum_Signal'] = 0
        df.loc[(df['RSI'] < 30) & (df['MACD'] > df['MACD_Signal']), 'Momentum_Signal'] = 1
        df.loc[(df['RSI'] > 70) & (df['MACD'] < df['MACD_Signal']), 'Momentum_Signal'] = -1

        # 3. Volatilitäts-Signale
        bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['Volatility_Signal'] = 0
        df.loc[(bb_width < bb_width.rolling(20).mean()) & (
                    df['ATR'] < df['ATR'].rolling(20).mean()), 'Volatility_Signal'] = 1
        df.loc[(bb_width > bb_width.rolling(20).mean() * 1.5), 'Volatility_Signal'] = -1

        # Kombinierte Signale mit Gewichtung
        df['Combined_Score'] = (
                self.weights.get('trend', 0) * df['Trend_Signal'] +
                self.weights.get('momentum', 0) * df['Momentum_Signal'] +
                self.weights.get('volatility', 0) * df['Volatility_Signal']
        )

        # Finale Signale basierend auf Schwellenwert
        df['Signal'] = 0
        df.loc[df['Combined_Score'] > self.threshold, 'Signal'] = 1
        df.loc[df['Combined_Score'] < -self.threshold, 'Signal'] = -1

        return df