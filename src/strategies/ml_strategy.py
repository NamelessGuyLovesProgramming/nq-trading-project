import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy
from src.utils.helpers import prepare_dataframe


class MLStrategy(BaseStrategy):
    def __init__(self, model, scaler, window_size=60, threshold=0.005):
        """
        Strategie basierend auf ML-Modellvorhersagen.

        Parameters:
        -----------
        model : Trainieres Modell
            Das trainierte ML-Modell für Vorhersagen
        scaler : sklearn.preprocessing.Scaler
            Der zum Skalieren der Daten verwendete Scaler
        window_size : int
            Größe des Zeitfensters für Eingabedaten
        threshold : float
            Schwellenwert für Signale
        """
        super().__init__(name="ML_Strategy")
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        self.threshold = threshold

    def prepare_features(self, data):
        """
        Bereitet Features für das Modell vor.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame mit OHLCV- und Indikatordaten

        Returns:
        --------
        np.array
            Eingabedaten für das Modell
        """
        # Daten für die Verarbeitung vorbereiten
        data = prepare_dataframe(data.copy())

        # Feature-Auswahl (muss mit Training übereinstimmen)
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_20', 'EMA_9', 'RSI', 'MACD', 'MACD_Hist',
                    'BB_Upper', 'BB_Lower', 'STOCH_k', 'ATR']

        feature_data = data[features].values

        # Skalieren der Daten
        feature_data = self.scaler.transform(feature_data)

        # Erstelle Sequenzen
        X = []
        for i in range(len(feature_data) - self.window_size + 1):
            X.append(feature_data[i:i + self.window_size])

        return np.array(X)

    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf Modellvorhersagen.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame mit OHLCV- und Indikatordaten

        Returns:
        --------
        pd.DataFrame
            DataFrame mit zusätzlicher Signalspalte
        """
        # Kopie erstellen und für die Verarbeitung vorbereiten
        df = prepare_dataframe(data.copy())

        # Bereite Features vor
        X = self.prepare_features(df)

        if len(X) == 0:
            print("Nicht genügend Daten für Vorhersagen.")
            df['Signal'] = 0
            return df

        # Vorhersagen
        predictions = self.model.predict(X).flatten()

        # Erstelle Signale df (mit Offset für Window)
        signals_df = pd.DataFrame(index=df.index[self.window_size - 1:])
        signals_df['Close'] = df['Close'].values[self.window_size - 1:]
        signals_df['Prediction'] = predictions

        # Berechne prozentuale Änderung von Tag zu Tag in den Vorhersagen
        signals_df['Pred_Change'] = signals_df['Prediction'].pct_change()
        signals_df.loc[signals_df.index[0], 'Pred_Change'] = 0  # Ersten NaN-Wert ersetzen

        # Generiere Signale
        signals_df['Signal'] = 0
        signals_df.loc[signals_df['Pred_Change'] > self.threshold, 'Signal'] = 1  # Kaufsignal
        signals_df.loc[signals_df['Pred_Change'] < -self.threshold, 'Signal'] = -1  # Verkaufssignal

        # Füge die Signale zum ursprünglichen DataFrame hinzu
        result_df = df.copy()
        result_df['Signal'] = 0
        result_df['Prediction'] = float('nan')  # Initialisiere mit NaN

        # Indizes verarbeiten - Verwende loc für zuverlässige und warnungsfreie Zuweisungen
        for idx, row in signals_df.iterrows():
            if idx in result_df.index:
                result_df.loc[idx, 'Signal'] = row['Signal']
                result_df.loc[idx, 'Prediction'] = row['Prediction']

        return result_df