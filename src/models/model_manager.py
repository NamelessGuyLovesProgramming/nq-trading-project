import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.data.processor import DataProcessor
from src.models.lstm import LSTMModel


class ModelManager:
    """
    Klasse zur Verwaltung von ML-Modellen, einschließlich Training, Speicherung und Laden.
    Unterstützt die Verwaltung verschiedener Feature-Sets und die GUI-Integration.
    """

    def __init__(self, models_dir='output/models'):
        """
        Initialisiert den ModelManager.

        Parameters:
        -----------
        models_dir : str
            Verzeichnis zum Speichern von Modellen
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        # Verzeichnis für Modell-Metadaten
        self.metadata_dir = os.path.join(models_dir, 'metadata')
        os.makedirs(self.metadata_dir, exist_ok=True)

        # Aktuelle Modelle und Scaler im Speicher
        self.models = {}
        self.scalers = {}

    def train_model(self, data, model_name, selected_features=None, window_size=60,
                    epochs=50, batch_size=32, test_size=0.2, data_source=None):
        """
        Trainiert ein neues Modell mit den angegebenen Parametern und Features.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame mit OHLCV-Daten und Indikatoren
        model_name : str
            Name des Modells (für Speicherung)
        selected_features : list
            Liste der ausgewählten Features für das Training
        window_size : int
            Größe des Zeitfensters für LSTM-Input
        epochs : int
            Anzahl der Trainingsepochen
        batch_size : int
            Batch-Größe für Training
        test_size : float
            Anteil der Daten für Testset
        data_source : dict
            Informationen über die Datenquelle (Symbol, Zeitraum, etc.)

        Returns:
        --------
        dict
            Dictionary mit Trainingsmetriken und Modellinfo
        """
        # Sicherstellen, dass der Modellname keine ungültigen Zeichen enthält
        model_name = self._sanitize_name(model_name)

        # Modellpfade festlegen
        model_file = os.path.join(self.models_dir, f"{model_name}.h5")
        metadata_file = os.path.join(self.metadata_dir, f"{model_name}.json")

        # DataProcessor für Datenvorbereitung
        processor = DataProcessor()

        # Daten mit ausgewählten Features vorbereiten
        X_train, y_train, X_test, y_test, scaler = processor.prepare_data_for_ml(
            data,
            window_size=window_size,
            test_size=test_size,
            selected_features=selected_features
        )

        # Modell erstellen
        input_shape = (window_size, X_train.shape[2])
        model = LSTMModel(input_shape, output_dir=self.models_dir)

        # Verwende die verbesserte Modellarchitektur
        model.build_improved_model()

        # Training mit Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                verbose=1,
                min_lr=0.00005
            )
        ]

        # Modell trainieren
        history = model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        # Modell speichern
        model.model.save(model_file)

        scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)

        # Vorhersagen und Metriken berechnen
        predictions = model.predict(X_test)

        # Stelle sicher, dass beide Arrays die gleiche Form haben
        predictions = predictions.flatten()
        y_test_flat = y_test.flatten()

        # Stelle sicher, dass beide Arrays die gleiche Länge haben
        min_len = min(len(predictions), len(y_test_flat))
        predictions = predictions[:min_len]
        y_test_flat = y_test_flat[:min_len]

        # Berechne MSE direkt
        squared_errors = (predictions - y_test_flat) ** 2
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)

        # Berechne MAE direkt
        mae = np.mean(np.abs(predictions - y_test_flat))

        # Berechne MAPE mit Vermeidung von Division durch Null
        non_zero_mask = y_test_flat != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_test_flat[non_zero_mask] - predictions[non_zero_mask]) /
                                  y_test_flat[non_zero_mask])) * 100
        else:
            mape = float('nan')

        # Metadaten erstellen
        metadata = {
            "name": model_name,
            "created_at": datetime.now().isoformat(),
            "window_size": window_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "input_shape": list(input_shape),
            "features": selected_features,
            "metrics": {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape) if not np.isnan(mape) else None
            },
            "training_samples": int(X_train.shape[0]),
            "testing_samples": int(X_test.shape[0]),
            "data_source": data_source  # Neue Information über Datenquellen
        }

        # Metadaten speichern
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Scaler im Speicher behalten
        self.models[model_name] = model.model
        self.scalers[model_name] = scaler

        return metadata

    # In src/models/model_manager.py, ändere die load_model-Methode:

    def load_model(self, model_name):
        """
        Lädt ein Modell und seinen Scaler.

        Parameters:
        -----------
        model_name : str
            Name des zu ladenden Modells

        Returns:
        --------
        tuple
            (model, scaler, metadata) oder (None, None, None) wenn nicht gefunden
        """
        model_name = self._sanitize_name(model_name)
        model_file = os.path.join(self.models_dir, f"{model_name}.h5")
        metadata_file = os.path.join(self.metadata_dir, f"{model_name}.json")

        # Prüfe, ob Dateien existieren
        if not os.path.exists(model_file) or not os.path.exists(metadata_file):
            print(f"Modell {model_name} nicht gefunden.")
            return None, None, None

        # Lade Modell, wenn nicht bereits im Speicher
        if model_name not in self.models:
            try:
                # Lade das Keras-Modell
                self.models[model_name] = tf.keras.models.load_model(model_file)
            except Exception as e:
                print(f"Fehler beim Laden des Modells: {e}")
                return None, None, None

        # Lade Metadaten
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Fehler beim Laden der Metadaten: {e}")
            metadata = None

        # Erstelle einen neuen Scaler und TRAINIERE IHN MIT DUMMY-DATEN
        # Dies ist der kritische Teil - wir müssen den Scaler mit etwas trainieren
        if model_name not in self.scalers:
            scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
            else:
                # Fallback mit Dummy-Daten
                self.scalers[model_name] = MinMaxScaler()
                feature_count = len(metadata.get('features', [])) if metadata else 5
                dummy_data = np.zeros((10, feature_count))
                self.scalers[model_name].fit(dummy_data)

        return self.models[model_name], self.scalers[model_name], metadata

    def list_available_models(self):
        """
        Listet alle verfügbaren Modelle mit Metadaten auf.

        Returns:
        --------
        list
            Liste von Dictionaries mit Modellmetadaten
        """
        models = []

        # Durchsuche das Metadaten-Verzeichnis nach JSON-Dateien
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.metadata_dir, filename), 'r') as f:
                        metadata = json.load(f)
                        # Prüfe, ob das Modell existiert
                        model_name = metadata.get('name', filename.replace('.json', ''))
                        model_file = os.path.join(self.models_dir, f"{model_name}.h5")

                        if os.path.exists(model_file):
                            models.append(metadata)
                except Exception as e:
                    print(f"Fehler beim Laden der Metadaten für {filename}: {e}")

        return models

    def delete_model(self, model_name):
        """
        Löscht ein Modell und alle zugehörigen Dateien (Metadaten, Scaler, etc.).

        Parameters:
        -----------
        model_name : str
            Name des zu löschenden Modells

        Returns:
        --------
        bool
            True wenn erfolgreich, False wenn fehlgeschlagen
        """
        model_name = self._sanitize_name(model_name)
        model_file = os.path.join(self.models_dir, f"{model_name}.h5")
        metadata_file = os.path.join(self.metadata_dir, f"{model_name}.json")
        scaler_file = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")

        # Suche auch nach anderen zugehörigen Dateien mit dem Modellnamen
        related_files = []
        for file in os.listdir(self.models_dir):
            if file.startswith(model_name + "_") or file == f"{model_name}.h5":
                related_files.append(os.path.join(self.models_dir, file))

        success = True
        files_deleted = []

        # Lösche Modelldatei
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
                files_deleted.append(model_file)
            except Exception as e:
                print(f"Fehler beim Löschen der Modelldatei: {e}")
                success = False

        # Lösche Metadatendatei
        if os.path.exists(metadata_file):
            try:
                os.remove(metadata_file)
                files_deleted.append(metadata_file)
            except Exception as e:
                print(f"Fehler beim Löschen der Metadatendatei: {e}")
                success = False

        # Lösche Scaler-Datei
        if os.path.exists(scaler_file):
            try:
                os.remove(scaler_file)
                files_deleted.append(scaler_file)
            except Exception as e:
                print(f"Fehler beim Löschen der Scaler-Datei: {e}")
                success = False

        # Lösche alle anderen zugehörigen Dateien
        for file in related_files:
            if os.path.exists(file) and file not in files_deleted:
                try:
                    os.remove(file)
                    files_deleted.append(file)
                except Exception as e:
                    print(f"Fehler beim Löschen der zugehörigen Datei {file}: {e}")
                    success = False

        # Entferne aus dem Speicher
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.scalers:
            del self.scalers[model_name]

        print(f"Modell {model_name} gelöscht. Erfolg: {success}. Gelöschte Dateien: {', '.join(files_deleted)}")
        return success

    def get_model_info(self, model_name):
        """
        Gibt detaillierte Informationen zu einem Modell zurück.

        Parameters:
        -----------
        model_name : str
            Name des Modells

        Returns:
        --------
        dict
            Metadaten des Modells oder None wenn nicht gefunden
        """
        model_name = self._sanitize_name(model_name)
        metadata_file = os.path.join(self.metadata_dir, f"{model_name}.json")

        if not os.path.exists(metadata_file):
            return None

        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Fehler beim Laden der Modellinfo: {e}")
            return None

    def _sanitize_name(self, name):
        """
        Bereinigt einen Modellnamen für die Verwendung als Dateiname.

        Parameters:
        -----------
        name : str
            Original-Modellname

        Returns:
        --------
        str
            Bereinigter Name
        """
        # Ersetze ungültige Zeichen durch Unterstriche
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
            name = name.replace(char, '_')

        return name
