import numpy as np
import pandas as pd
from src.strategies.base import BaseStrategy
from src.utils.helpers import prepare_dataframe


class MLStrategy(BaseStrategy):
    def __init__(self, model, scaler, window_size=60, threshold=0.005, selected_features=None):
        """
        Strategie basierend auf ML-Modellvorhersagen.

        Parameters:
        -----------
        model : Trainiertes Modell
            Das trainierte ML-Modell für Vorhersagen
        scaler : sklearn.preprocessing.Scaler
            Der zum Skalieren der Daten verwendete Scaler
        window_size : int
            Größe des Zeitfensters für Eingabedaten
        threshold : float
            Schwellenwert für Signale
        selected_features : list
            Liste der für das Modell ausgewählten Features.
            Wenn None, wird versucht, die Features aus dem Scaler zu extrahieren,
            oder eine Standardliste wird verwendet.
        """
        super().__init__(name="ML_Strategy")
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        self.threshold = threshold

        # Versuche, die Anzahl der Features aus dem Scaler zu erhalten
        n_features = getattr(scaler, 'n_features_in_', None)

        if selected_features is not None:
            # Verwende die übergebene Feature-Liste
            self.selected_features = selected_features

            # Überprüfe, ob die Anzahl der Features mit dem Scaler übereinstimmt
            if n_features is not None and len(self.selected_features) != n_features:
                print(
                    f"Warnung: Anzahl der ausgewählten Features ({len(self.selected_features)}) "
                    f"stimmt nicht mit der vom Scaler erwarteten Anzahl ({n_features}) überein."
                )
        elif n_features is not None:
            # Wenn keine Features angegeben wurden, aber wir die Anzahl kennen,
            # verwende eine passende Standardliste
            standard_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'EMA_9', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Middle', 'BB_Upper', 'BB_Lower', 'STOCH_k', 'STOCH_d',
                'ATR', 'OBV'
            ]

            # Verwende so viele Features wie benötigt
            if n_features <= len(standard_features):
                self.selected_features = standard_features[:n_features]
            else:
                # Falls wir mehr Features benötigen als in der Standardliste,
                # fülle mit Dummy-Namen auf
                self.selected_features = standard_features + [
                    f"Feature_{i}" for i in range(len(standard_features), n_features)
                ]

            print(f"Features automatisch aus Scaler extrahiert: {self.selected_features}")
        else:
            # Fallback auf Standardliste
            self.selected_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'EMA_9', 'RSI', 'MACD', 'BB_Middle', 'BB_Upper', 'BB_Lower'
            ]
            print(f"Verwende Standardliste von {len(self.selected_features)} Features")
    def prepare_features(self, data):
        """
        Bereitet Features für das Modell vor und stellt sicher, dass die Dimensionen
        mit denen des trainierten Scalers übereinstimmen.

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

        # Anzahl der vom Scaler erwarteten Features ermitteln
        expected_features_count = getattr(self.scaler, 'n_features_in_', len(self.selected_features))

        # Prüfe, ob die ausgewählten Features verfügbar sind
        available_features = [f for f in self.selected_features if f in data.columns]
        missing_features = set(self.selected_features) - set(available_features)

        # Wenn nicht alle Features verfügbar sind, gib eine Warnung aus
        if missing_features:
            print(f"Warnung: Folgende Features fehlen: {missing_features}")
            print(f"Verfügbare Features: {list(data.columns)}")

            # Erstelle Daten-Frame mit fehlenden Spalten (gefüllt mit 0)
            for missing_feature in missing_features:
                data[missing_feature] = 0

            # Aktualisiere die verfügbaren Features
            available_features = self.selected_features

        # Extrahiere die Features in der richtigen Reihenfolge
        feature_data = data[available_features].values

        # Prüfe, ob die Dimensionen passen
        actual_features = feature_data.shape[1]

        if expected_features_count != actual_features:
            # Wenn die Anzahl nicht passt, fülle mit 0-Spalten auf
            padding_needed = expected_features_count - actual_features
            if padding_needed > 0:
                print(f"Fülle {padding_needed} fehlende Feature-Spalten mit 0 auf")
                padding = np.zeros((feature_data.shape[0], padding_needed))
                feature_data = np.hstack([feature_data, padding])
            else:
                raise ValueError(
                    f"Feature-Dimensionen stimmen nicht überein. "
                    f"Scaler erwartet {expected_features_count} Features, aber {actual_features} wurden bereitgestellt. "
                    f"Diese Diskrepanz kann nicht automatisch behoben werden."
                )

        # Skaliere die Daten mit dem vorhandenen Scaler
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