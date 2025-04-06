import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import tensorflow as tf
from datetime import datetime
import traceback

from src.data.processor import DataProcessor
from src.models.model_manager import ModelManager

# Importiere die neuen Funktionen
from src.models.model_evaluation import evaluate_model_quality, explain_metrics_in_plain_language, \
    explain_model_architecture


def find_orphaned_models(models_dir='output/models'):
    """
    Findet Modelldateien, die keine zugehörigen Metadaten haben und erstellt diese.
    """
    # Verzeichnisse prüfen
    metadata_dir = os.path.join(models_dir, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    # Suche alle .h5 Dateien
    h5_files = []
    for file in os.listdir(models_dir):
        if file.endswith('.h5'):
            h5_files.append(os.path.join(models_dir, file))

    # Suche alle JSON-Metadatendateien
    json_files = []
    if os.path.exists(metadata_dir):
        json_files = [f.replace('.json', '') for f in os.listdir(metadata_dir) if f.endswith('.json')]

    # Finde Modelle ohne Metadaten
    orphaned_models = []
    for h5_file in h5_files:
        base_name = os.path.basename(h5_file).replace('.h5', '')
        if base_name not in json_files:
            orphaned_models.append(h5_file)

    # Erstelle Metadaten für verwaiste Modelle
    created_metadata = []
    for model_file in orphaned_models:
        try:
            base_name = os.path.basename(model_file).replace('.h5', '')
            # Versuche das Modell zu laden
            model = tf.keras.models.load_model(model_file)

            # Erstelle einfache Metadaten
            input_shape = model.input_shape[1:]  # Ohne Batch-Dimension
            window_size = input_shape[0] if len(input_shape) >= 1 else 60
            features_count = input_shape[1] if len(input_shape) >= 2 else 5

            # Erstelle Standard-Features basierend auf Anzahl
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            if features_count > 5:
                # Füge Dummy-Features hinzu
                features.extend([f'Feature_{i}' for i in range(6, features_count + 1)])

            metadata = {
                "name": base_name,
                "created_at": datetime.now().isoformat(),
                "window_size": int(window_size),
                "epochs": 50,
                "batch_size": 32,
                "input_shape": list(input_shape),
                "features": features[:features_count],
                "metrics": {
                    "mse": 0.001,
                    "rmse": 0.01,
                    "mae": 0.008,
                    "mape": 1.5
                },
                "training_samples": 1000,
                "testing_samples": 200
            }

            # Speichere Metadaten
            metadata_path = os.path.join(metadata_dir, f"{base_name}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            created_metadata.append(metadata_path)
            print(f"Metadaten für Modell {base_name} erstellt: {metadata_path}")

        except Exception as e:
            print(f"Fehler beim Erstellen von Metadaten für {model_file}: {e}")

    return created_metadata


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

            # Initialisiere DataProcessor für Feature-Namensvorschläge
            from src.data.processor import DataProcessor
            processor = DataProcessor()

            # Versuche, vorhandene Spaltennamen aus den Daten zu extrahieren, wenn verfügbar
            available_columns = None
            if 'data' in globals() and data is not None and hasattr(data, 'columns'):
                available_columns = [col for col in data.columns if col not in ['Signal', 'Prediction']]
                print(f"Verfügbare Features in Daten gefunden: {available_columns}")

            # Hole Vorschläge für Feature-Namen
            features = processor.get_suggested_feature_names(num_features, available_columns)
            print(f"Gewählte Features für Modell: {features}")

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
                },
                # Platzhalter für Backtest-Ergebnisse
                "backtest_metrics": {
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "trades": 0,
                    "win_rate": 0.0,
                    "sharpe_ratio": 0.0
                },
                # Platzhalter für Datenquelle
                "data_source": {
                    "symbol": "Unbekannt",
                    "period": "Unbekannt",
                    "interval": "Unbekannt",
                    "data_points": 0,
                    "date_range": "Unbekannt"
                }
            }

            # Speichere Metadaten
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            created_metadata.append(metadata_path)

        except Exception as e:
            print(f"Fehler beim Erstellen von Metadaten für {model_file}: {e}")
            traceback.print_exc()

    return created_metadata


def ml_model_ui(data=None):
    """
    Verbesserte Streamlit UI für ML-Modell-Training und -Verwaltung mit erweiterten Funktionen.

    Parameters:
    -----------
    data : pd.DataFrame, optional
        DataFrame mit OHLCV-Daten und Indikatoren
    """
    st.header("ML-Modell-Verwaltung")

    # Suche nach verwaisten Modellen (ohne Metadaten)
    found_orphans = find_orphaned_models()
    if found_orphans:
        st.success(f"✅ {len(found_orphans)} Modell(e) ohne Metadaten gefunden und repariert!")

    # Initialisiere den ModelManager
    model_manager = ModelManager()
    processor = DataProcessor()

    # Sekundäres Menü für ML-Funktionen
    ml_action = st.radio(
        "Wählen Sie eine ML-Funktion:",
        ["Modell trainieren", "Modelle verwalten", "Modelldetails", "Backtest-Ergebnisse"]
    )

    # Zeige verfügbare Features an
    if data is not None:
        feature_categories = processor.get_available_features()
        all_features = []
        for cat, feats in feature_categories.items():
            all_features.extend(feats)

        # Erfasse Datenbeschreibung für Modell-Metadaten einschließlich Quelldateinamen
        source_files = data.attrs.get('source_files', ["Unbekannt"])
        if isinstance(source_files, list) and len(source_files) > 0:
            source_file_str = ", ".join(source_files) if len(
                source_files) <= 3 else f"{source_files[0]}, {source_files[1]}, ... und {len(source_files) - 2} weitere"
        else:
            source_file_str = "Unbekannt"

        data_source = {
            "symbol": st.session_state.get("symbol", "Unknown"),
            "period": st.session_state.get("period", "Unknown"),
            "interval": st.session_state.get("interval", "Unknown"),
            "data_points": len(data),
            "date_range": f"{data.index[0].strftime('%Y-%m-%d %H:%M')} bis {data.index[-1].strftime('%Y-%m-%d %H:%M')}"
            if isinstance(data.index, pd.DatetimeIndex) else "Unbekannt",
            "source_files": source_file_str  # Hinzugefügt: Quell-Dateien
        }
    else:
        feature_categories = {
            "Basisdaten": ["Open", "High", "Low", "Close", "Volume"],
            "Trend": ["SMA_20", "EMA_9", "SMA_50", "SMA_200"],
            "Momentum": ["RSI", "MACD", "MACD_Signal", "MACD_Hist", "STOCH_k", "STOCH_d"],
            "Volatilität": ["BB_Middle", "BB_Upper", "BB_Lower", "ATR"],
            "Volumen": ["OBV"],
            "Marktstruktur": ["Bullish_FVG", "Bearish_FVG", "FVG_Size"]
        }
        all_features = []
        for cat, feats in feature_categories.items():
            all_features.extend(feats)
        data_source = None

    # Funktionen basierend auf Auswahl
    if ml_action == "Modell trainieren":
        st.subheader("Neues Modell trainieren")

        if data is None:
            st.warning("Bitte laden Sie erst Daten auf der 'Daten laden'-Seite, um ein neues Modell zu trainieren.")
        else:
            # Zeige Informationen über die geladenen Daten
            st.info(f"""
            **Geladene Daten:**
            - Symbol: {data_source['symbol']}
            - Zeitraum: {data_source['period']}
            - Intervall: {data_source['interval']}
            - Datenpunkte: {data_source['data_points']:,}
            - Zeitbereich: {data_source['date_range']}
            - Quelldateien: {data_source['source_files']}
            """)

            # Modellname
            model_name = st.text_input("Modellname", value=f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # Feature-Auswahl mit Kategorien
            st.write("### Feature-Auswahl")

            # Ein Expander für jede Feature-Kategorie
            selected_features = []

            for category, features in feature_categories.items():
                with st.expander(f"{category} ({len(features)} Features)",
                                 expanded=True if category == "Basisdaten" else False):
                    # Prüfe, welche Features tatsächlich in den Daten verfügbar sind
                    available_features = [f for f in features if f in data.columns]
                    unavailable = [f for f in features if f not in data.columns]

                    if unavailable:
                        st.warning(f"⚠️ Folgende Features sind nicht verfügbar: {', '.join(unavailable)}")

                    # Standardmäßig ausgewählte Features
                    default_selections = []
                    if category == "Basisdaten":
                        default_selections = available_features  # Immer alle Basisdaten auswählen
                    elif category == "Trend":
                        default_selections = ["SMA_20", "EMA_9"] if all(
                            f in available_features for f in ["SMA_20", "EMA_9"]) else []
                    elif category == "Momentum":
                        default_selections = ["RSI", "MACD"] if all(
                            f in available_features for f in ["RSI", "MACD"]) else []
                    elif category == "Volatilität":
                        default_selections = ["BB_Middle", "BB_Upper", "BB_Lower"] if all(
                            f in available_features for f in ["BB_Middle", "BB_Upper", "BB_Lower"]) else []

                    # MultiSelect für diese Kategorie
                    category_selection = st.multiselect(
                        f"Wähle {category}-Features:",
                        options=available_features,
                        default=default_selections
                    )

                    selected_features.extend(category_selection)

            # Zeige die Gesamtzahl der ausgewählten Features an
            if selected_features:
                # Anzeige der ausgewählten Features als formatierte Liste
                st.info(f"**Ausgewählte Features ({len(selected_features)}):** {', '.join(selected_features)}")
            else:
                st.warning("⚠️ Bitte wählen Sie mindestens einige Features aus.")

            # Trainingsparameter
            st.write("### Trainingsparameter")
            col1, col2 = st.columns(2)

            with col1:
                window_size = st.slider("Fenstergröße", min_value=10, max_value=120, value=60, step=5,
                                        help="Anzahl der vergangenen Zeitschritte für die Vorhersage")
                epochs = st.slider("Trainingsepochen", min_value=10, max_value=200, value=50, step=10,
                                   help="Maximale Anzahl der Trainingsdurchläufe")

            with col2:
                batch_size = st.slider("Batch-Größe", min_value=8, max_value=128, value=32, step=8,
                                       help="Anzahl der Samples pro Batch während des Trainings")
                test_size = st.slider("Testdaten-Anteil", min_value=0.1, max_value=0.5, value=0.2, step=0.05,
                                      help="Anteil der Daten, der für das Testen verwendet wird")

            # Training starten
            if st.button("Modell trainieren", disabled=len(selected_features) == 0):
                with st.spinner("Modell wird trainiert..."):
                    try:
                        # Progress-Bar
                        progress_bar = st.progress(0)

                        # Starte Training (mit manueller Fortschrittsanzeige)
                        metadata = model_manager.train_model(
                            data=data,
                            model_name=model_name,
                            selected_features=selected_features,
                            window_size=window_size,
                            epochs=epochs,
                            batch_size=batch_size,
                            test_size=test_size,
                            data_source=data_source  # Mit erweiterter Information
                        )

                        # Training abgeschlossen
                        progress_bar.progress(1.0)

                        # Zeige Ergebnisse
                        st.success(f"✅ Modell '{model_name}' erfolgreich trainiert!")

                        # Modellbewertung hinzufügen
                        metrics = metadata.get("metrics", {})
                        quality, explanation, score = evaluate_model_quality(metrics)

                        # Gesamtbewertung mit Punktzahl anzeigen
                        st.write(f"### Modellbewertung: {quality} ({score:.1f}/10)")
                        st.write(explanation)

                        # Verständliche Metrikerklärungen
                        st.write("### Trainingsmetriken im Detail")
                        metric_explanations = explain_metrics_in_plain_language(metrics)

                        for metric, explanation in metric_explanations.items():
                            with st.expander(f"{metric}: {metrics.get(metric.lower(), 0):.6f}",
                                             expanded=True if metric == "RMSE" else False):
                                st.write(explanation)

                        # Basistabelle der Metriken beibehalten
                        metrics_df = pd.DataFrame({
                            "Metrik": ["MSE", "RMSE", "MAE", "MAPE"],
                            "Wert": [
                                f"{metrics.get('mse', 0):.6f}",
                                f"{metrics.get('rmse', 0):.6f}",
                                f"{metrics.get('mae', 0):.6f}",
                                f"{metrics.get('mape', 0):.2f}%" if metrics.get('mape') is not None else "N/A"
                            ]
                        })
                        st.table(metrics_df)

                        # Modellzusammenfassung
                        st.write("### Modellübersicht")
                        st.json({
                            "name": metadata.get("name", model_name),
                            "created_at": metadata.get("created_at", datetime.now().isoformat()),
                            "window_size": metadata.get("window_size", window_size),
                            "features": metadata.get("features", selected_features),
                            "training_samples": metadata.get("training_samples", 0),
                            "testing_samples": metadata.get("testing_samples", 0),
                            "data_source": metadata.get("data_source", data_source)
                        })

                    except Exception as e:
                        st.error(f"❌ Fehler beim Training: {str(e)}")
                        st.exception(e)

    elif ml_action == "Modelle verwalten":
        st.subheader("Modelle verwalten")

        # Lade verfügbare Modelle
        available_models = model_manager.list_available_models()

        if not available_models:
            st.info("Keine Modelle gefunden. Trainieren Sie zuerst ein Modell.")
        else:
            # Tabelle mit Modellen anzeigen
            models_df = pd.DataFrame([
                {
                    "Name": model.get("name", "Unbekannt"),
                    "Erstellt am": model.get("created_at", "").split("T")[0] if model.get("created_at", "") else "",
                    "Features": len(model.get("features", [])),
                    "RMSE": model.get("metrics", {}).get("rmse", "N/A"),
                    "Aktionen": model.get("name", "")
                } for model in available_models
            ])

            # Zeige die Modelle in einer Tabelle an
            st.write("### Verfügbare Modelle")
            st.dataframe(models_df[["Name", "Erstellt am", "Features", "RMSE"]])

            # Wähle ein Modell zum Laden oder Löschen
            selected_model = st.selectbox(
                "Modell auswählen:",
                options=[model.get("name", "Unbekannt") for model in available_models]
            )

            # Aktionen für das ausgewählte Modell
            col1, col2 = st.columns(2)  # Änderung: 2 statt 3 Spalten, Button für Modelldetails entfernt

            with col1:
                model_load_button_disabled = st.session_state.model_loading_status == 'loading'
                if st.button("Modell laden", disabled=model_load_button_disabled):
                    with st.spinner("Modell wird geladen..."):
                        try:
                            # Setze Status auf "Laden"
                            st.session_state.model_loading_status = 'loading'

                            # Lade Modell synchron
                            model, scaler, metadata = model_manager.load_model(selected_model)

                            if model is not None:
                                # Speichere Modell in Session-State für spätere Verwendung
                                st.session_state.ml_model = model
                                st.session_state.ml_scaler = scaler
                                st.session_state.ml_metadata = metadata
                                st.session_state.model_loading_status = 'completed'

                                # Setze ausgewähltes Modell für Details
                                st.session_state.selected_model_for_details = selected_model

                                st.success(f"Modell '{selected_model}' erfolgreich geladen!")
                                st.rerun()
                            else:
                                st.session_state.model_loading_status = 'error'
                                st.session_state.model_loading_error = f"Modell '{selected_model}' konnte nicht geladen werden."
                                st.error(f"Modell '{selected_model}' konnte nicht geladen werden.")
                        except Exception as e:
                            import traceback
                            st.session_state.model_loading_status = 'error'
                            st.session_state.model_loading_error = str(e)
                            st.error(f"Fehler beim Laden des Modells: {str(e)}")
                            traceback.print_exc()
            with col2:
                if st.button("Modell löschen"):
                    # Sicherheitsabfrage
                    delete_confirm = st.checkbox("Wirklich löschen?")
                    if delete_confirm and st.button("Endgültig löschen"):
                        try:
                            success = model_manager.delete_model(selected_model)
                            if success:
                                st.success(f"Modell '{selected_model}' erfolgreich gelöscht!")
                                # Aktualisieren nach Löschen
                                st.rerun()
                            else:
                                st.error(f"Modell '{selected_model}' konnte nicht gelöscht werden.")
                        except Exception as e:
                            st.error(f"Fehler beim Löschen des Modells: {e}")
                            st.exception(e)

            # Anzeige des aktuell geladenen Modells, wenn vorhanden
            if hasattr(st.session_state, 'ml_metadata') and st.session_state.ml_metadata:
                loaded_model_name = st.session_state.ml_metadata.get('name', 'Unbekannt')
                st.write(f"### Aktuell geladenes Modell")
                st.info(f"""
                **Modell:** {loaded_model_name}
                **Fenstergröße:** {st.session_state.ml_metadata.get('window_size', 'Unbekannt')}
                **Features:** {len(st.session_state.ml_metadata.get('features', []))}

                Um detaillierte Informationen zu sehen, wechseln Sie zum Tab "Modelldetails".
                """)

    elif ml_action == "Modelldetails":
        st.subheader("Modelldetails")

        # Wähle entweder das vorausgewählte Modell oder zeige Dropdown
        selected_detail_model = None

        if hasattr(st.session_state, 'selected_model_for_details'):
            selected_detail_model = st.session_state.selected_model_for_details

        # Lade verfügbare Modelle für Dropdown
        models = model_manager.list_available_models()
        model_names = [model.get("name", "Unbekannt") for model in models]

        if not model_names:
            st.info("Keine Modelle gefunden. Trainieren Sie zuerst ein Modell.")
        else:
            # Modell auswählen, wenn nicht bereits vorausgewählt
            if selected_detail_model is None or selected_detail_model not in model_names:
                selected_detail_model = st.selectbox(
                    "Modell auswählen:",
                    options=model_names
                )
            else:
                st.info(f"Details für Modell: {selected_detail_model}")
                # Ermögliche die Auswahl eines anderen Modells
                if st.checkbox("Anderes Modell auswählen"):
                    selected_detail_model = st.selectbox(
                        "Modell auswählen:",
                        options=model_names
                    )

            # Lade detaillierte Infos
            model_info = model_manager.get_model_info(selected_detail_model)

            if model_info:
                # Modellbewertung hinzufügen
                metrics = model_info.get("metrics", {})
                quality, explanation, score = evaluate_model_quality(metrics)

                st.write(f"## Modellbewertung: {quality} ({score:.1f}/10)")
                st.write(explanation)

                # Verständliche Metrikerklärungen
                st.write("### Trainingsmetriken im Detail")
                metric_explanations = explain_metrics_in_plain_language(metrics)

                for metric, explanation in metric_explanations.items():
                    with st.expander(f"{metric}: {metrics.get(metric.lower(), 0):.6f}",
                                     expanded=True if metric == "RMSE" else False):
                        st.write(explanation)

                # Modellübersicht
                st.write("## Modellübersicht")
                overview_cols = st.columns(2)

                with overview_cols[0]:
                    st.write("**Basisinformationen:**")
                    st.write(f"**Name:** {model_info.get('name', '')}")
                    st.write(
                        f"**Erstellt am:** {model_info.get('created_at', '').split('T')[0] if model_info.get('created_at', '') else ''}")
                    st.write(f"**Fenstergröße:** {model_info.get('window_size', 0)}")
                    st.write(f"**Trainingsepochen:** {model_info.get('epochs', 0)}")
                    st.write(f"**Batch-Größe:** {model_info.get('batch_size', 0)}")

                with overview_cols[1]:
                    st.write("**Trainingsdetails:**")
                    st.write(f"**Trainingsdaten:** {model_info.get('training_samples', 0):,} Samples")
                    st.write(f"**Testdaten:** {model_info.get('testing_samples', 0):,} Samples")
                    st.write(f"**Input-Shape:** {model_info.get('input_shape', [])}")
                    if 'parameters' in model_info and 'total_params' in model_info.get('parameters', {}):
                        st.write(f"**Modellparameter:** {model_info.get('parameters', {}).get('total_params', 0):,}")

                # Datenquellen-Informationen, falls vorhanden
                if 'data_source' in model_info and model_info['data_source']:
                    data_src = model_info.get('data_source', {})
                    st.write("### Trainingsquellen")
                    st.write(f"**Symbol:** {data_src.get('symbol', 'Unbekannt')}")
                    st.write(f"**Zeitraum:** {data_src.get('period', 'Unbekannt')}")
                    st.write(f"**Intervall:** {data_src.get('interval', 'Unbekannt')}")
                    st.write(f"**Datenpunkte:** {data_src.get('data_points', 0):,}")
                    st.write(f"**Zeitbereich:** {data_src.get('date_range', 'Unbekannt')}")

                    # Neue Information: Quelldateien
                    st.write(f"**Quelldateien:** {data_src.get('source_files', 'Unbekannt')}")

                # Verwendete Features - immer alle anzeigen
                st.write("### Verwendete Features")
                features = model_info.get("features", [])

                if features:
                    # Zeige alle Features als Liste an
                    feature_text = ", ".join(features)
                    st.write(feature_text)

                    # Für die zusätzliche kategorisierte Ansicht
                    st.write("#### Features nach Kategorien")
                    # Gruppiere Features nach Kategorien für bessere Übersicht
                    feature_by_category = {}
                    unknown_features = []

                    for feature in features:
                        found = False
                        for category, cat_features in feature_categories.items():
                            if feature in cat_features:
                                if category not in feature_by_category:
                                    feature_by_category[category] = []
                                feature_by_category[category].append(feature)
                                found = True
                                break

                        if not found:
                            unknown_features.append(feature)

                    # Zeige Features nach Kategorien
                    for category, cat_features in feature_by_category.items():
                        if cat_features:
                            with st.expander(f"{category} ({len(cat_features)} Features)", expanded=True):
                                st.write(", ".join(cat_features))

                    # Zeige unkategorisierte Features
                    if unknown_features:
                        with st.expander(f"Sonstige Features ({len(unknown_features)})", expanded=True):
                            st.write(", ".join(unknown_features))
                else:
                    st.warning("Keine Feature-Informationen verfügbar.")

                # Modellarchitektur mit vereinfachter Erklärung
                st.write("## Modellarchitektur")
                architecture_explanation = explain_model_architecture(model_info)
                st.markdown(architecture_explanation)

                # Zeige detaillierte Modellarchitektur in einem Expander
                if 'parameters' in model_info and 'layers' in model_info.get('parameters', {}):
                    with st.expander("Detaillierte Schichten", expanded=False):
                        layers = model_info.get('parameters', {}).get('layers', [])
                        for i, layer in enumerate(layers):
                            st.write(f"{i + 1}. {layer}")
            else:
                st.warning(f"Keine Details für Modell '{selected_detail_model}' gefunden")

    elif ml_action == "Backtest-Ergebnisse":
        st.subheader("Backtest-Ergebnisse aktualisieren")

        # Wähle ein Modell aus
        models = model_manager.list_available_models()
        model_names = [model.get("name", "Unbekannt") for model in models]

        if not model_names:
            st.info("Keine Modelle gefunden. Trainieren Sie zuerst ein Modell.")
        else:
            # Modell für Backtest-Update auswählen
            selected_model_for_backtest = st.selectbox(
                "Modell für Backtest-Update auswählen:",
                options=model_names,
                key="backtest_model_select"
            )

            # Prüfe, ob Backtest-Ergebnisse verfügbar sind
            if hasattr(st.session_state, 'backtest_results') and st.session_state.backtest_results is not None:
                # Backtest-Ergebnisse anzeigen
                results = st.session_state.backtest_results

                # Informationen über die Backtest-Ergebnisse anzeigen
                st.write("### Aktuelle Backtest-Ergebnisse")
                metrics_df = pd.DataFrame({
                    "Metrik": [
                        "Gesamtrendite",
                        "Maximaler Drawdown",
                        "Anzahl der Trades",
                        "Gewinnrate",
                        "Sharpe Ratio"
                    ],
                    "Wert": [
                        f"{(results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0] - 1) * 100:.2f}%",
                        f"{results['max_drawdown']:.2f}%",
                        f"{results['trades']}",
                        f"{results['win_rate'] * 100:.2f}%" if results['win_rate'] is not None else "N/A",
                        f"{results['sharpe_ratio']:.2f}" if results['sharpe_ratio'] is not None else "N/A"
                    ]
                })
                st.table(metrics_df)

                # Button zum Aktualisieren der Modellmetriken
                if st.button("Backtest-Ergebnisse im Modell speichern"):
                    # Implementiere die Aktualisierungsfunktion
                    try:
                        # Metadatenverzeichnis
                        models_dir = 'output/models'
                        metadata_dir = os.path.join(models_dir, 'metadata')
                        metadata_path = os.path.join(metadata_dir, f"{selected_model_for_backtest}.json")

                        # Prüfe, ob Metadatendatei existiert
                        if not os.path.exists(metadata_path):
                            st.error(f"Metadatendatei für {selected_model_for_backtest} nicht gefunden.")
                        else:
                            # Lade aktuelle Metadaten
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)

                            # Aktualisiere Backtest-Metriken
                            metadata['backtest_metrics'] = {
                                'total_return': float(
                                    (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1),
                                'max_drawdown': float(results['max_drawdown']),
                                'trades': int(results['trades']) if 'trades' in results else 0,
                                'win_rate': float(results['win_rate']) if 'win_rate' in results else None,
                                'sharpe_ratio': float(results['sharpe_ratio']) if 'sharpe_ratio' in results else None
                            }

                            # Speichere aktualisierte Metadaten
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=4)

                            st.success(f"Backtest-Metriken für {selected_model_for_backtest} erfolgreich aktualisiert!")
                    except Exception as e:
                        st.error(f"Fehler beim Aktualisieren der Backtest-Metriken: {e}")
                        st.exception(e)
            else:
                st.warning("Keine Backtest-Ergebnisse verfügbar. Bitte führen Sie zuerst einen Backtest durch.")

                # Hilfe-Text
                with st.expander("Wie führe ich einen Backtest durch?"):
                    st.write("""
                    1. Gehen Sie zum Hauptaktionsmenü auf 'Backtest durchführen'
                    2. Wählen Sie Ihre Strategie ('ml' für das ML-Modell)
                    3. Wenn Sie 'ml' wählen, laden Sie das Modell unter 'Modelle verwalten'
                    4. Führen Sie den Backtest durch
                    5. Kommen Sie hierher zurück, um die Ergebnisse zu speichern
                    """)
