import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self):
        """
        Initialisiert den DataProcessor.
        """
        self.scaler = MinMaxScaler()

    def add_technical_indicators(self, df):
        """
        Fügt technische Indikatoren zum DataFrame hinzu mit reinen Pandas-Funktionen.
        Verbesserte Fehlerbehandlung und Datenvorbereitung.
        """
        # Kopie erstellen und sicherstellen, dass alle erforderlichen Spalten vorhanden sind
        ohlc_df = df.copy()
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Überprüfe erforderliche Spalten
        for col in required_cols:
            if col not in ohlc_df.columns:
                print(f"Warnung: Spalte '{col}' nicht gefunden!")

                # Versuche alternative Spaltennamen
                alt_col = col.lower()
                if alt_col in ohlc_df.columns:
                    ohlc_df[col] = ohlc_df[alt_col]
                    print(f"  Verwende '{alt_col}' stattdessen.")
                else:
                    print(f"  Kann '{col}' nicht finden. Indikatorberechnung könnte fehlschlagen!")

        # 'Adj Close' in 'Adj_Close' umbenennen, falls vorhanden
        if 'Adj Close' in ohlc_df.columns:
            ohlc_df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)

        # Fehlende Werte füllen für robustere Indikatorberechnung
        # Zuerst linear interpolieren
        ohlc_df = ohlc_df.infer_objects(copy=False).interpolate(method='linear')
        # Versuchen Sie diese Lösung
        # Konvertiere erst numerische Spalten
        for col in ohlc_df.select_dtypes(include=['float64', 'int64']).columns:
            ohlc_df[col] = ohlc_df[col].interpolate(method='linear')
        # Dann behandle andere Spalten separat, falls nötig
        for col in ohlc_df.select_dtypes(exclude=['float64', 'int64']).columns:
            if col not in ['Date', 'Time']:  # Ignoriere Datums-/Zeitspalten
                try:
                    ohlc_df[col] = pd.to_numeric(ohlc_df[col], errors='coerce')
                    ohlc_df[col] = ohlc_df[col].interpolate(method='linear')
                except:
                    pass  # Ignoriere Fehler für nicht-numerische Spalten

        # Dann verbleibende NaNs mit forward/backward fill behandeln
        # Verwende ffill und bfill statt fillna(method=...)
        ohlc_df = ohlc_df.ffill().bfill()

        print(f"DataFrame-Form nach Vorverarbeitung: {ohlc_df.shape}")

        try:
            # SMA - Simple Moving Average
            ohlc_df['SMA_9'] = ohlc_df['Close'].rolling(window=9).mean()
            ohlc_df['SMA_20'] = ohlc_df['Close'].rolling(window=20).mean()
            ohlc_df['SMA_50'] = ohlc_df['Close'].rolling(window=50).mean()
            ohlc_df['SMA_200'] = ohlc_df['Close'].rolling(window=200).mean()

            # EMA - Exponential Moving Average
            ohlc_df['EMA_9'] = ohlc_df['Close'].ewm(span=9, adjust=False).mean()
            ohlc_df['EMA_20'] = ohlc_df['Close'].ewm(span=20, adjust=False).mean()

            # MACD - Moving Average Convergence Divergence
            ema12 = ohlc_df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = ohlc_df['Close'].ewm(span=26, adjust=False).mean()
            ohlc_df['MACD'] = ema12 - ema26
            ohlc_df['MACD_Signal'] = ohlc_df['MACD'].ewm(span=9, adjust=False).mean()
            ohlc_df['MACD_Hist'] = ohlc_df['MACD'] - ohlc_df['MACD_Signal']

            # RSI - Relative Strength Index
            delta = ohlc_df['Close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            ohlc_df['RSI'] = 100 - (100 / (1 + rs))
            # Behandle Extremfälle bei RSI-Berechnung
            ohlc_df['RSI'] = ohlc_df['RSI'].clip(0, 100)

            # Bollinger Bands
            rolling_mean = ohlc_df['Close'].rolling(window=20).mean()
            rolling_std = ohlc_df['Close'].rolling(window=20).std()
            ohlc_df['BB_Middle'] = rolling_mean
            ohlc_df['BB_Upper'] = rolling_mean + (2 * rolling_std)
            ohlc_df['BB_Lower'] = rolling_mean - (2 * rolling_std)

            # Stochastic Oscillator (vereinfacht)
            n = 14
            high_n = ohlc_df['High'].rolling(window=n).max()
            low_n = ohlc_df['Low'].rolling(window=n).min()
            ohlc_df['STOCH_k'] = 100 * ((ohlc_df['Close'] - low_n) / (high_n - low_n))
            ohlc_df['STOCH_d'] = ohlc_df['STOCH_k'].rolling(window=3).mean()
            # Behandle potentielle Division durch Null
            ohlc_df['STOCH_k'] = ohlc_df['STOCH_k'].fillna(50)
            ohlc_df['STOCH_d'] = ohlc_df['STOCH_d'].fillna(50)

            # ATR - Average True Range
            high_low = ohlc_df['High'] - ohlc_df['Low']
            high_close = (ohlc_df['High'] - ohlc_df['Close'].shift()).abs()
            low_close = (ohlc_df['Low'] - ohlc_df['Close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            ohlc_df['ATR'] = true_range.rolling(14).mean()

            # OBV - On Balance Volume mit verbesserter Fehlerbehandlung
            close_array = ohlc_df['Close'].values
            volume_array = ohlc_df['Volume'].values
            obv_array = np.zeros_like(volume_array)

            # Sichere OBV-Berechnung mit NumPy und Fehlerprüfung
            for i in range(1, len(close_array)):
                if np.isnan(close_array[i]) or np.isnan(close_array[i - 1]) or np.isnan(volume_array[i]):
                    # Bei NaN-Werten einfach den vorherigen OBV-Wert beibehalten
                    obv_array[i] = obv_array[i - 1]
                else:
                    if close_array[i] > close_array[i - 1]:
                        obv_array[i] = obv_array[i - 1] + volume_array[i]
                    elif close_array[i] < close_array[i - 1]:
                        obv_array[i] = obv_array[i - 1] - volume_array[i]
                    else:
                        obv_array[i] = obv_array[i - 1]

            ohlc_df['OBV'] = obv_array

            # Überprüfe auf NaN-Werte in den berechneten Indikatoren
            nan_counts = ohlc_df.isna().sum()
            if nan_counts.sum() > 0:
                print("Warnung: Es gibt NaN-Werte in den berechneten Indikatoren:")
                for col, count in nan_counts.items():
                    if count > 0:
                        print(f"  {col}: {count} NaN-Werte")

                # Fülle verbleibende NaN-Werte in den Indikatoren
                # Verwende ffill und bfill statt fillna(method=...)
                ohlc_df = ohlc_df.ffill().bfill()
                print("NaN-Werte wurden gefüllt.")

        except Exception as e:
            print(f"Fehler bei der Indikatorberechnung: {e}")
            # Trotz Fehler das teilweise gefüllte DataFrame zurückgeben

        print(f"Indikatoren erfolgreich hinzugefügt. DataFrame-Form: {ohlc_df.shape}")
        return ohlc_df

    def prepare_data_for_ml(self, df, target_col='Close', window_size=60, test_size=0.2, scale=True):
        """
        Bereitet Daten für das ML-Modell vor.
        """
        df_ml = df.copy()

        # Debug-Information vor dem Entfernen von NaN-Werten
        print(f"DataFrame vor NaN-Entfernung: {df_ml.shape}")

        # Fehlende Werte füllen statt komplett zu entfernen
        # Für numerische Spalten, verwende lineare Interpolation
        df_ml = df_ml.interpolate(method='linear')

        # Für verbleibende NaN-Werte (z.B. am Anfang der Zeitreihe)
        # Verwende ffill und bfill statt fillna(method=...)
        df_ml = df_ml.ffill().bfill()

        print(f"DataFrame nach NaN-Behandlung: {df_ml.shape}")

        # Feature-Auswahl (kann angepasst werden)
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_20', 'EMA_9', 'RSI', 'MACD', 'MACD_Hist',
                    'BB_Upper', 'BB_Lower', 'STOCH_k', 'ATR']

        # Prüfe, ob alle Features vorhanden sind
        available_features = [f for f in features if f in df_ml.columns]
        if len(available_features) < len(features):
            missing = set(features) - set(available_features)
            print(f"Warnung: Folgende Features fehlen: {missing}")
            features = available_features

        print(f"Verwendete Features: {features}")

        # Extrahiere Feature- und Zieldaten
        feature_data = df_ml[features].values
        target_data = df_ml[target_col].values

        print(f"Feature-Daten Shape: {feature_data.shape}")
        print(f"Ziel-Daten Shape: {target_data.shape}")

        # Skalierung
        if scale and feature_data.shape[0] > 0:
            feature_data = self.scaler.fit_transform(feature_data)
            print("Daten wurden skaliert")

        # Erstelle Sequenzen für LSTM
        X, y = [], []
        for i in range(len(feature_data) - window_size):
            X.append(feature_data[i:i + window_size])
            y.append(target_data[i + window_size])

        X, y = np.array(X), np.array(y)
        print(f"X Shape nach Sequenzerstellung: {X.shape}")
        print(f"y Shape nach Sequenzerstellung: {y.shape}")

        # Split in Trainings- und Testdaten
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        print(f"X_train Shape: {X_train.shape}, X_test Shape: {X_test.shape}")

        return X_train, y_train, X_test, y_test, self.scaler

    def generate_signals(self, df, predictions, threshold=0.01):
        """
        Generiert Signale basierend auf Vorhersagen.
        """
        signals_df = df.copy()
        signals_df = signals_df.iloc[-len(predictions):]

        # Berechne prozentuale Änderung
        predicted_changes = np.zeros_like(predictions)
        predicted_changes[1:] = (predictions[1:] - predictions[:-1]) / predictions[:-1]

        # Generiere Signale
        signals = np.zeros_like(predicted_changes)
        signals[predicted_changes > threshold] = 1  # Kaufsignal
        signals[predicted_changes < -threshold] = -1  # Verkaufssignal

        signals_df['Prediction'] = predictions
        signals_df['Signal'] = signals

        return signals_df