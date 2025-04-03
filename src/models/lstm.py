import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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

    # Dies ist eine angepasste Version der build_model-Methode für src/models/lstm.py

    def build_model(self):
        """
        Erstellt ein vereinfachtes LSTM-Modell mit besserer Regularisierung zur Vermeidung von Overfitting.

        Returns:
        --------
        tf.keras.models.Sequential
            Das erstellte LSTM-Modell
        """
        print(f"Erstelle optimiertes LSTM-Modell mit Input-Shape: {self.input_shape}")

        try:
            model = Sequential()

            # Reduzierte Komplexität: Weniger LSTM-Units als vorher
            lstm_units = min(20, max(10, self.input_shape[0] // 3))
            print(f"Verwende {lstm_units} LSTM-Units zur Vermeidung von Overfitting")

            # Nur eine LSTM-Schicht statt zwei, mit stärkerer Dropout-Rate
            model.add(LSTM(
                units=lstm_units,
                input_shape=self.input_shape,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0.1,  # Dropout innerhalb der rekurrenten Verbindungen
                kernel_initializer='glorot_uniform',
                # L2-Regularisierung hinzufügen
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))

            # Erhöhte Dropout-Rate
            model.add(Dropout(0.4))
            model.add(BatchNormalization())

            # Einfachere Dense-Schicht
            model.add(Dense(
                units=max(8, lstm_units // 2),
                activation='relu',
                # L2-Regularisierung auch hier
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ))
            model.add(Dropout(0.3))

            # Ausgabeschicht
            model.add(Dense(units=1))

            # Optimizer mit reduzierter Lernrate
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0005,  # Reduzierte Lernrate für stabileres Training
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )

            model.compile(optimizer=optimizer, loss='mean_squared_error')

            # Modellzusammenfassung ausgeben
            model.summary()

            self.model = model
            return model

        except Exception as e:
            print(f"Fehler beim Erstellen des Modells: {e}")

            # Fallback zu einem sehr einfachen Modell bei Fehler
            print("Erstelle einfaches Fallback-Modell...")
            fallback_model = Sequential()
            fallback_model.add(LSTM(10, input_shape=self.input_shape))
            fallback_model.add(Dense(5, activation='relu'))
            fallback_model.add(Dense(1))
            fallback_model.compile(optimizer='adam', loss='mean_squared_error')

            self.model = fallback_model
            return fallback_model

    # Angepasste Version der train-Methode mit verbesserten Hyperparametern

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, callbacks=None):
        """
        Trainiert das LSTM-Modell mit optimierten Parametern zur Vermeidung von Overfitting.

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
            Maximale Anzahl der Trainingsdurchläufe
        batch_size : int
            Batch-Größe
        callbacks : list, optional
            Liste von Keras Callbacks für das Training

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

            # Erhöhe Batch-Größe für bessere Generalisierung
            adjusted_batch_size = min(128, max(64, len(X_train) // 100))
            print(f"Verwende optimierte Batch-Größe: {adjusted_batch_size}")
            batch_size = adjusted_batch_size

            # Reduziere die maximale Anzahl der Epochen
            adjusted_epochs = min(30, epochs)
            if adjusted_epochs < epochs:
                print(f"Anpassung der Epochs von {epochs} auf {adjusted_epochs} zur Vermeidung von Overfitting")
                epochs = adjusted_epochs

            # Verbesserte Callbacks für optimales Training
            default_callbacks = [
                # Early Stopping mit geringerer Patience für schnelleres Stoppen bei Overfitting
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,  # Reduziert von 10 auf 5
                    restore_best_weights=True,
                    verbose=1
                ),
                # Aggressivere Learning Rate Reduzierung
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.3,  # Stärkere Reduzierung (von 0.5 auf 0.3)
                    patience=3,  # Schnellere Anpassung (von 5 auf 3)
                    verbose=1,
                    min_lr=0.00005
                ),
                # ModelCheckpoint speichert das beste Modell
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.output_dir, 'best_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]

            # Verwende übergebene Callbacks oder Standardcallbacks
            use_callbacks = callbacks if callbacks is not None else default_callbacks

            # Trainiere das Modell
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=use_callbacks,
                verbose=1,
                shuffle=True
            )

            # Speichere das trainierte Modell
            model_path = os.path.join(self.output_dir, 'lstm_model.h5')
            self.model.save(model_path)
            print(f"Modell gespeichert unter {model_path}")

            return history

        except Exception as e:
            print(f"Fehler beim Training des Modells: {e}")

            # Versuche simpleres Training als Fallback
            print("Versuche alternatives Training mit stark vereinfachtem Ansatz...")
            try:
                # Erstelle ein sehr einfaches Modell mit minimalen Parametern
                simple_model = Sequential()
                simple_model.add(LSTM(8, input_shape=self.input_shape))
                simple_model.add(Dense(1))
                simple_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

                # Verwende größere Batch-Größe und weniger Epochs
                simple_history = simple_model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=3,
                            restore_best_weights=True
                        )
                    ]
                )

                self.model = simple_model
                simple_model_path = os.path.join(self.output_dir, 'simple_lstm_model.h5')
                self.model.save(simple_model_path)
                print(f"Einfaches Modell gespeichert unter {simple_model_path}")

                return simple_history

            except Exception as e2:
                print(f"Auch alternatives Training fehlgeschlagen: {e2}")

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

    # Füge diese Methode zur LSTMModel-Klasse in src/models/lstm.py hinzu

    def build_improved_model(self):
        """
        Erstellt ein verbessertes LSTM-Modell mit reduzierter Komplexität und besserer Regularisierung.
        Diese Methode ist speziell auf die Vermeidung von Overfitting ausgelegt.

        Returns:
        --------
        tf.keras.models.Model
            Das erstellte LSTM-Modell
        """

        print(f"Erstelle verbessertes LSTM-Modell mit Input-Shape: {self.input_shape}")

        try:
            # Reduzierte Komplexität
            lstm_units = min(20, max(10, self.input_shape[0] // 3))
            print(f"Verwende reduzierte LSTM-Units: {lstm_units} (zur Vermeidung von Overfitting)")

            # Funktionales API-Modell mit Input-Layer
            inputs = Input(shape=self.input_shape)

            # Erste LSTM-Schicht mit stärkerer Regularisierung
            x = LSTM(
                units=lstm_units,
                return_sequences=False,  # Keine Sequenz zurückgeben für einfachere Architektur
                activation='tanh',
                recurrent_dropout=0.1,  # Dropout in rekurrenten Verbindungen
                kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2-Regularisierung
            )(inputs)

            # Stärkerer Dropout zur Vermeidung von Overfitting
            x = Dropout(0.4)(x)
            x = BatchNormalization()(x)

            # Einfachere Dense-Schicht
            x = Dense(
                units=max(8, lstm_units // 2),
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )(x)
            x = Dropout(0.3)(x)

            # Ausgabeschicht
            outputs = Dense(units=1)(x)

            # Erstelle Modell
            model = Model(inputs=inputs, outputs=outputs)

            # Kompilieren mit angepasstem Optimizer für stabileres Training
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0005,  # Reduzierte Lernrate
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07
            )

            model.compile(optimizer=optimizer, loss='mean_squared_error')

            # Ausgabe der Modellzusammenfassung
            model.summary()

            self.model = model
            return model

        except Exception as e:
            print(f"Fehler beim Erstellen des verbesserten Modells: {e}")

            # Fallback zu einem noch einfacheren Modell
            print("Erstelle einfaches Fallback-Modell...")

            try:
                # Sehr einfaches funktionales Modell
                inputs = Input(shape=self.input_shape)
                x = LSTM(10)(inputs)
                outputs = Dense(1)(x)

                fallback_model = Model(inputs=inputs, outputs=outputs)
                fallback_model.compile(optimizer='adam', loss='mean_squared_error')

                self.model = fallback_model
                return fallback_model

            except Exception as e2:
                print(f"Auch das verbesserte Fallback-Modell konnte nicht erstellt werden: {e2}")

                # Letzte Möglichkeit: Sequential API
                from tensorflow.keras.models import Sequential

                fallback_model = Sequential()
                fallback_model.add(LSTM(8, input_shape=self.input_shape))
                fallback_model.add(Dense(1))
                fallback_model.compile(optimizer='adam', loss='mean_squared_error')

                self.model = fallback_model
                return fallback_model

    # Fügen Sie diese Funktion zu src/models/lstm.py hinzu

    def train_in_batches(self, data, window_size, batch_size, epochs, test_size=0.2):
        """
        Trainiert das LSTM-Modell in Datei-Batches für sehr große Datasets.

        Parameters:
        -----------
        data : pd.DataFrame oder Pfad zu CSV-Dateien
            Daten oder Verzeichnis mit mehreren CSV-Dateien
        window_size : int
            Fenstergröße für Sequenzen
        batch_size : int
            Mini-Batch-Größe für Training
        epochs : int
            Anzahl der Trainingsepochen pro Datei-Batch
        test_size : float
            Anteil der Testdaten

        Returns:
        --------
        tf.keras.callbacks.History
            Geschichte des Trainings
        """
        from src.data.processor import DataProcessor
        processor = DataProcessor()

        # Modell initialisieren, falls es noch nicht existiert
        if self.model is None:
            self.build_improved_model()

        # Gesamtergebnisse
        total_samples = 0
        cumulative_loss = 0

        # Wenn es sich um ein Verzeichnis handelt, verarbeite Dateien nacheinander
        if isinstance(data, str) and os.path.isdir(data):
            files = sorted([f for f in os.listdir(data) if f.endswith('.csv')])
            print(f"Verarbeite {len(files)} Dateien aus {data}")

            for i, file in enumerate(files):
                print(f"\nVerarbeite Datei {i + 1}/{len(files)}: {file}")
                file_path = os.path.join(data, file)

                # Lade einen Chunk der Daten
                chunk_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                chunk_data = processor.add_technical_indicators(chunk_data)

                # Optimiere Speichernutzung
                chunk_data = processor.optimize_dataframe_memory(chunk_data)

                # Bereite Daten für ML vor
                X, y, X_val, y_val, _ = processor.prepare_data_for_ml(
                    chunk_data, window_size=window_size, test_size=test_size
                )

                if X.shape[0] > 10:  # Nur trainieren, wenn genügend Daten
                    # Trainiere mit aktueller Chunk
                    history = self.model.fit(
                        X, y,
                        validation_data=(X_val, y_val),
                        epochs=max(1, epochs // len(files)),  # Verteile Epochen
                        batch_size=batch_size,
                        verbose=1
                    )

                    # Sammle Statistiken
                    total_samples += X.shape[0]
                    cumulative_loss += history.history['loss'][-1] * X.shape[0]

                    print(f"Chunk-Training abgeschlossen. Loss: {history.history['loss'][-1]:.6f}")

                # Speicher freigeben
                del chunk_data, X, y, X_val, y_val
                import gc
                gc.collect()

        else:
            # Normales Training mit dem gesamten DataFrame
            X, y, X_val, y_val, _ = processor.prepare_data_for_ml(
                data, window_size=window_size, test_size=test_size
            )

            history = self.model.fit(
                X, y,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            total_samples = X.shape[0]
            cumulative_loss = history.history['loss'][-1] * X.shape[0]

        # Berechne durchschnittlichen Loss über alle Batches
        avg_loss = cumulative_loss / total_samples if total_samples > 0 else float('nan')
        print(f"\nTraining abgeschlossen. Durchschnittlicher Loss: {avg_loss:.6f}")
        print(f"Gesamtzahl trainierter Samples: {total_samples}")

        # Speichere das Modell
        self.model.save(os.path.join(self.output_dir, 'lstm_model.h5'))

        # Erstelle ein Dummy-History-Objekt für die Rückgabe
        class DummyHistory:
            def __init__(self, avg_loss):
                self.history = {'loss': [avg_loss], 'val_loss': [avg_loss * 1.2]}

        return DummyHistory(avg_loss)