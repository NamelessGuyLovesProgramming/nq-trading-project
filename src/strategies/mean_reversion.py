import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.utils.helpers import prepare_dataframe


class MeanReversionStrategy(BaseStrategy):
    def __init__(self, rsi_overbought=70, rsi_oversold=30, bb_trigger=0.8):
        """
        Mean-Reversion-Strategie basierend auf RSI und Bollinger Bands.

        Parameters:
        -----------
        rsi_overbought : int
            RSI-Wert für überkaufte Bedingung
        rsi_oversold : int
            RSI-Wert für überverkaufte Bedingung
        bb_trigger : float
            Wie weit der Preis in Richtung BB-Bänder gehen muss (0 bis 1)
        """
        super().__init__(name="Mean_Reversion")
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_trigger = bb_trigger

    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf Mean-Reversion-Logik.
        """
        # Daten für die Verarbeitung vorbereiten
        df = prepare_dataframe(data.copy())
        df['Signal'] = 0

        # Handle potential DataFrames instead of Series
        if isinstance(df['Close'], pd.DataFrame):
            close = df['Close'].iloc[:, 0]
        else:
            close = df['Close']

        if isinstance(df['BB_Lower'], pd.DataFrame):
            bb_lower = df['BB_Lower'].iloc[:, 0]
        else:
            bb_lower = df['BB_Lower']

        if isinstance(df['BB_Upper'], pd.DataFrame):
            bb_upper = df['BB_Upper'].iloc[:, 0]
        else:
            bb_upper = df['BB_Upper']

        if isinstance(df['RSI'], pd.DataFrame):
            rsi = df['RSI'].iloc[:, 0]
        else:
            rsi = df['RSI']

        # Berechne Bollinger Band Position (0 = untere Bande, 1 = obere Bande)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        df['BB_Position'] = bb_position

        # Überverkauft: RSI < oversold UND Preis nahe unterem BB
        oversold = (rsi < self.rsi_oversold) & (bb_position < self.bb_trigger)

        # Überkauft: RSI > overbought UND Preis nahe oberem BB
        overbought = (rsi > self.rsi_overbought) & (bb_position > (1 - self.bb_trigger))

        # Generiere Signale
        df.loc[oversold, 'Signal'] = 1  # Kaufsignal bei überverkauft
        df.loc[overbought, 'Signal'] = -1  # Verkaufssignal bei überkauft

        return df