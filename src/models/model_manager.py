import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

from src.data.processor import DataProcessor
from src.models.lstm import LSTMModel
from web.ml_model_ui import scan_for_orphaned_models


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
                   epochs=50, batch_size=32, test_size=0.2):
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

        # Vorhersagen und Metriken berechnen
        predictions = model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))

        # Berechne MAPE mit Vermeidung von Division durch Null
        non_zero_mask = y_test != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
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
            "testing_samples": int(X_test.shape[0])
        }

        # Metadaten speichern
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Scaler im Speicher behalten
        self.models[model_name] = model.model
        self.scalers[model_name] = scaler

        return metadata

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

        # Erstelle einen neuen Scaler (da wir den originalen nicht speichern können)
        # Hierfür brauchen wir Trainingsdaten - dies ist ein Kompromiss
        # In einer vollständigen Implementierung würden wir den Scaler serialisieren
        if model_name not in self.scalers:
            self.scalers[model_name] = MinMaxScaler()

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
        Löscht ein Modell und seine Metadaten.

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

        success = True

        # Lösche Modelldatei
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
            except Exception as e:
                print(f"Fehler beim Löschen der Modelldatei: {e}")
                success = False

        # Lösche Metadatendatei
        if os.path.exists(metadata_file):
            try:
                os.remove(metadata_file)
            except Exception as e:
                print(f"Fehler beim Löschen der Metadatendatei: {e}")
                success = False

        # Entferne aus dem Speicher
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.scalers:
            del self.scalers[model_name]

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

    import os
    import json
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from datetime import datetime
    from sklearn.preprocessing import MinMaxScaler

    # Diese Funktion kann in die ModelManager-Klasse integriert werden
    def scan_for_orphaned_models(models_dir='output/models'):
        """
        Scannt nach Modellen ohne Metadatendateien und erstellt einfache Metadaten für sie.

        Returns:
        --------
        list
            Liste der neu erstellten Metadatendateien
        """
        # Metadatenverzeichnis
        metadata_dir = os.path.join(models_dir, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)

        # Suche nach .h5 Dateien
        h5_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]

        # Suche nach JSON-Dateien
        json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        json_basenames = [os.path.splitext(f)[0] for f in json_files]

        # Finde Modelle ohne Metadaten
        orphaned_models = [f for f in h5_files if os.path.splitext(f)[0] not in json_basenames]

        # Erstelle Metadaten für verwaiste Modelle
        created_metadata = []

        for model_file in orphaned_models:
            try:
                model_path = os.path.join(models_dir, model_file)
                metadata_path = os.path.join(metadata_dir, os.path.splitext(model_file)[0] + '.json')

                # Lade das Modell
                model = tf.keras.models.load_model(model_path)

                # Extrahiere Modellinformationen
                input_shape = model.input_shape[1:]  # Entferne Batch-Dimension

                # Bestimme window_size aus dem Input-Shape
                window_size = input_shape[0] if len(input_shape) > 1 else input_shape[0]

                # Extrahiere Anzahl der Features
                num_features = input_shape[1] if len(input_shape) > 1 else 1

                # Erstelle Standardfeatures basierend auf Anzahl
                standard_features = ['Open', 'High', 'Low', 'Close', 'Volume']
                technical_indicators = [
                    'SMA_20', 'EMA_9', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                    'BB_Upper', 'BB_Lower', 'BB_Middle', 'ATR', 'STOCH_k', 'STOCH_d'
                ]

                if num_features <= 5:
                    features = standard_features[:num_features]
                else:
                    features = standard_features + technical_indicators[:num_features - 5]

                # Generiere Metadaten
                metadata = {
                    "name": os.path.splitext(model_file)[0],
                    "created_at": datetime.now().isoformat(),
                    "window_size": window_size,
                    "epochs": 50,  # Standardwert, kann später aktualisiert werden
                    "batch_size": 32,  # Standardwert, kann später aktualisiert werden
                    "input_shape": list(input_shape),
                    "features": features,
                    "metrics": {
                        "mse": 0.0001,  # Platzhalterwerte, die später aktualisiert werden können
                        "rmse": 0.01,
                        "mae": 0.008,
                        "mape": 1.5
                    },
                    "training_samples": 1000,  # Platzhalter
                    "testing_samples": 200,  # Platzhalter
                    "parameters": {
                        "layers": [layer.name for layer in model.layers],
                        "total_params": model.count_params()
                    }
                }

                # Speichere Metadaten
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                print(f"Metadaten für {model_file} erstellt: {metadata_path}")
                created_metadata.append(metadata_path)

            except Exception as e:
                print(f"Fehler beim Erstellen von Metadaten für {model_file}: {e}")

        return created_metadata

    # Die folgenden Methoden können in die ModelManager-Klasse eingefügt oder aktualisiert werden

    def update_model_metrics(model_name, backtest_results, models_dir='output/models'):
        """
        Aktualisiert die Metriken eines vorhandenen Modells mit Backtest-Ergebnissen.

        Parameters:
        -----------
        model_name : str
            Name des Modells (ohne .h5-Erweiterung)
        backtest_results : dict
            Dictionary mit Backtest-Ergebnissen
        models_dir : str
            Verzeichnis der Modelle

        Returns:
        --------
        bool
            True wenn erfolgreich, False wenn fehlgeschlagen
        """
        # Metadatenverzeichnis
        metadata_dir = os.path.join(models_dir, 'metadata')
        metadata_path = os.path.join(metadata_dir, f"{model_name}.json")

        # Prüfe, ob Metadatendatei existiert
        if not os.path.exists(metadata_path):
            print(f"Metadatendatei für {model_name} nicht gefunden.")
            return False

        try:
            # Lade aktuelle Metadaten
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Extrahiere relevante Metriken aus den Backtest-Ergebnissen
            if 'max_drawdown' in backtest_results:
                metadata['backtest_metrics'] = {
                    'total_return': float((backtest_results['portfolio_value'].iloc[-1] /
                                           backtest_results['portfolio_value'].iloc[0]) - 1),
                    'max_drawdown': float(backtest_results['max_drawdown']),
                    'trades': int(backtest_results['trades']) if 'trades' in backtest_results else 0,
                    'win_rate': float(backtest_results['win_rate']) if 'win_rate' in backtest_results else None,
                    'sharpe_ratio': float(
                        backtest_results['sharpe_ratio']) if 'sharpe_ratio' in backtest_results else None
                }

                # Speichere aktualisierte Metadaten
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                print(f"Backtest-Metriken für {model_name} aktualisiert")
                return True
            else:
                print("Ungültiges Backtest-Ergebnis-Format")
                return False

        except Exception as e:
            print(f"Fehler beim Aktualisieren der Modellmetriken: {e}")
            return False

    def update_model_features(model_name, features, models_dir='output/models'):
        """
        Aktualisiert die Features eines vorhandenen Modells.

        Parameters:
        -----------
        model_name : str
            Name des Modells (ohne .h5-Erweiterung)
        features : list
            Liste der Features, die das Modell verwendet
        models_dir : str
            Verzeichnis der Modelle

        Returns:
        --------
        bool
            True wenn erfolgreich, False wenn fehlgeschlagen
        """
        # Metadatenverzeichnis
        metadata_dir = os.path.join(models_dir, 'metadata')
        metadata_path = os.path.join(metadata_dir, f"{model_name}.json")

        # Prüfe, ob Metadatendatei existiert
        if not os.path.exists(metadata_path):
            print(f"Metadatendatei für {model_name} nicht gefunden.")
            return False

        try:
            # Lade aktuelle Metadaten
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Aktualisiere Features
            metadata['features'] = features

            # Speichere aktualisierte Metadaten
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            print(f"Features für {model_name} aktualisiert")
            return True

        except Exception as e:
            print(f"Fehler beim Aktualisieren der Modellfeatures: {e}")
            return False

    # Beispielcode für die Integration in die Streamlit-App
    def integrate_scan_for_models(self):
        """
        Integriert die scan_for_orphaned_models-Funktion in die Streamlit-App.
        """
        import streamlit as st

        if st.button("Nach unbeschriebenen Modellen suchen"):
            with st.spinner("Suche nach Modellen ohne Metadaten..."):
                created_files = scan_for_orphaned_models()
                if created_files:
                    st.success(f"{len(created_files)} Metadatendateien erstellt!")
                else:
                    st.info("Keine unbeschriebenen Modelle gefunden.")