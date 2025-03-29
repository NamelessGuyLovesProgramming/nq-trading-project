# src/strategies/volume_profile.py
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.utils.helpers import prepare_dataframe


class VolumeProfileStrategy(BaseStrategy):
    def __init__(self, volume_threshold=2.0, lookback=20):
        """
        Strategie basierend auf Volumenanomalien und Preisausbrüchen.

        Parameters:
        -----------
        volume_threshold : float
            Volumen-Schwellenwert als Vielfaches des Durchschnitts
        lookback : int
            Lookback-Periode für Preislevels
        """
        super().__init__(name="Volume_Profile")
        self.volume_threshold = volume_threshold
        self.lookback = lookback

    def generate_signals(self, data):
        # Daten für die Verarbeitung vorbereiten
        df = prepare_dataframe(data.copy())

        # Berechne relative Volumenveränderung
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        # Identifiziere signifikante Preisniveaus (Widerstände/Unterstützungen)
        df['Recent_High'] = df['High'].rolling(window=self.lookback).max()
        df['Recent_Low'] = df['Low'].rolling(window=self.lookback).min()

        # Signale
        df['Signal'] = 0

        # Kaufsignal: Ausbruch über Recent_High bei hohem Volumen
        buy_condition = (
                (df['Close'] > df['Recent_High'].shift(1)) &
                (df['Volume_Ratio'] > self.volume_threshold)
        )

        # Verkaufssignal: Ausbruch unter Recent_Low bei hohem Volumen
        sell_condition = (
                (df['Close'] < df['Recent_Low'].shift(1)) &
                (df['Volume_Ratio'] > self.volume_threshold)
        )

        df.loc[buy_condition, 'Signal'] = 1
        df.loc[sell_condition, 'Signal'] = -1

        return df