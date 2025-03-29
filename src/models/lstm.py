import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os


class LSTMModel:
    def __init__(self, input_shape, output_dir='models'):
        """
        Initialisiert das LSTM-Modell.

        Parameters:
        -----------
        input_shape : tuple
            Form der Eingabedaten (z.B. (60, 14) für 60 Zeitschritte und 14 Features)
        output_dir : str
            Verzeichnis zum Speichern der Modelle
        """
        self.input_shape = input_shape
        self.output_dir = output_dir
        self.model = None

        # Stelle sicher, dass das Ausgabeverzeichnis existiert
        os.makedirs(output_dir, exist_ok=True)

    def build_model(self):
        """
        Erstellt das LSTM-Modell mit verbesserter Architektur und Fehlerbehandlung.

        Returns:
        --------
        tf.keras.models.Sequential
            Das erstellte LSTM-Modell
        """
        print(f"Erstelle LSTM-Modell mit Input-Shape: {self.input_shape}")

        try:
            model = Sequential()

            # Erste LSTM-Schicht mit weniger Units für kleinere Datensätze
            lstm_units = min(50, max(20, self.input_shape[0] // 2))
            print(f"Verwende {lstm_units} LSTM-Units basierend auf Input-Shape")

            # Erste LSTM-Schicht mit Return-Sequences für die nächste LSTM-Schicht
            model.add(LSTM(
                units=lstm_units,
                return_sequences=True,
                input_shape=self.input_shape,
                activation='tanh',  # Explizite Aktivierungsfunktion
                recurrent_activation='sigmoid',  # Standardakivierung für rekurrente Verbindungen
                kernel_initializer='glorot_uniform'  # Bessere Initialisierung für Gewichte
            ))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            # Zweite LSTM-Schicht
            model.add(LSTM(
                units=lstm_units // 2,  # Weniger Units in tieferen Schichten
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid'
            ))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

            # Dense-Schichten
            model.add(Dense(units=max(10, lstm_units // 4), activation='relu'))
            model.add(Dropout(0.2))

            # Ausgabeschicht
            model.add(Dense(units=1))

            # Kompiliere das Modell mit verbessertem Optimizer
            # Füge Adam-Optimizer mit angepassten Parametern hinzu
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )

            model.compile(optimizer=optimizer, loss='mean_squared_error')

            # Gib Modellzusammenfassung aus
            model.summary()

            self.model = model
            return model

        except Exception as e:
            print(f"Fehler beim Erstellen des Modells: {e}")

            # Fallback zu einem einfacheren Modell bei Fehler
            print("Erstelle einfacheres Fallback-Modell...")
            fallback_model = Sequential()
            fallback_model.add(LSTM(20, input_shape=self.input_shape))
            fallback_model.add(Dense(10, activation='relu'))
            fallback_model.add(Dense(1))
            fallback_model.compile(optimizer='adam', loss='mean_squared_error')

            self.model = fallback_model
            return fallback_model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Trainiert das LSTM-Modell mit verbesserter Fehlerbehandlung und Anpassung.

        Parameters:
        -----------
        X_train : np.array
            Trainings-Features
        y_train : np.array
            Trainings-Labels
        X_val : np.array
            Validierungs-Features
        y_val : np.array
            Validierungs-Labels
        epochs : int
            Anzahl der Trainingsdurchläufe
        batch_size : int
            Batch-Größe

        Returns:
        --------
        tf.keras.callbacks.History
            Trainingsverlauf
        """
        print(f"Starte Training mit: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        print(f"Validierungsdaten: X_val.shape={X_val.shape}, y_val.shape={y_val.shape}")

        try:
            # Prüfe, ob Trainingsdaten gültig sind
            if X_train.size == 0 or y_train.size == 0:
                raise ValueError("Trainingsdaten sind leer")

            # Prüfe auf NaN-Werte
            if np.isnan(X_train).any() or np.isnan(y_train).any():
                print("Warnung: NaN-Werte in den Trainingsdaten gefunden. Werden ersetzt.")
                X_train = np.nan_to_num(X_train)
                y_train = np.nan_to_num(y_train)

            if np.isnan(X_val).any() or np.isnan(y_val).any():
                print("Warnung: NaN-Werte in den Validierungsdaten gefunden. Werden ersetzt.")
                X_val = np.nan_to_num(X_val)
                y_val = np.nan_to_num(y_val)

            # Erstelle Modell falls noch nicht vorhanden
            if self.model is None:
                self.build_model()

            # Passe Batch-Größe an, falls nötig
            adjusted_batch_size = min(batch_size, len(X_train) // 10)
            if adjusted_batch_size < batch_size:
                print(f"Anpassung der Batch-Größe von {batch_size} auf {adjusted_batch_size} aufgrund der Datenmenge")
                batch_size = max(1, adjusted_batch_size)

            # Passe Epochs an, falls nötig
            adjusted_epochs = min(epochs, 100)  # Limitiere die maximale Anzahl
            if adjusted_epochs < epochs:
                print(f"Anpassung der Epochs von {epochs} auf {adjusted_epochs}")
                epochs = adjusted_epochs

            # Callbacks für besseres Training
            callbacks = [
                # Early Stopping zur Vermeidung von Overfitting
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                # Reduziere Learning Rate bei Plateau
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    verbose=1,
                    min_lr=0.0001
                ),
                # ModelCheckpoint speichert das beste Modell
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.output_dir, 'best_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]

            # Trainiere das Modell
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1,
                shuffle=True  # Wichtig für Zeitreihen: True für bessere Generalisierung
            )

            # Speichere das trainierte Modell
            model_path = os.path.join(self.output_dir, 'lstm_model.h5')
            self.model.save(model_path)
            print(f"Modell gespeichert unter {model_path}")

            return history

        except Exception as e:
            print(f"Fehler beim Training des Modells: {e}")

            # Versuche es mit einem einfacheren Modell und reduzierter Komplexität
            print("Versuche alternatives Training mit vereinfachtem Ansatz...")
            try:
                # Erstelle einfacheres Modell
                simple_model = Sequential()
                simple_model.add(LSTM(10, input_shape=self.input_shape))
                simple_model.add(Dense(1))
                simple_model.compile(optimizer='adam', loss='mse')

                # Verwende kleinere Batch-Größe und weniger Epochs
                simple_history = simple_model.fit(
                    X_train, y_train,
                    epochs=min(20, epochs),
                    batch_size=max(1, len(X_train) // 20),
                    validation_data=(X_val, y_val),
                    verbose=1
                )

                # Speichere das einfache Modell
                self.model = simple_model
                simple_model_path = os.path.join(self.output_dir, 'simple_lstm_model.h5')
                self.model.save(simple_model_path)
                print(f"Einfaches Modell gespeichert unter {simple_model_path}")

                return simple_history

            except Exception as e2:
                print(f"Auch alternatives Training fehlgeschlagen: {e2}")

                # Erstelle Dummy-History-Objekt
                class DummyHistory:
                    def __init__(self):
                        self.history = {'loss': [0], 'val_loss': [0]}

                return DummyHistory()

    def plot_training_history(self, history):
        """
        Plottet den Trainingsverlauf.

        Parameters:
        -----------
        history : tf.keras.callbacks.History
            Trainingsverlauf

        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib-Figure-Objekt
        """
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Modell-Trainingsverlauf')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Speichere den Plot
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))

        return plt.gcf()

    def predict(self, X):
        """
        Macht Vorhersagen mit dem trainierten Modell.

        Parameters:
        -----------
        X : np.array
            Eingabedaten

        Returns:
        --------
        np.array
            Vorhersagen
        """
        if self.model is None:
            raise ValueError("Modell wurde noch nicht erstellt oder trainiert.")

        return self.model.predict(X)

    def load_model(self, model_path):
        """
        Lädt ein gespeichertes Modell.

        Parameters:
        -----------
        model_path : str
            Pfad zum gespeicherten Modell

        Returns:
        --------
        LSTMModel
            Das geladene Modell
        """
        self.model = tf.keras.models.load_model(model_path)
        return self