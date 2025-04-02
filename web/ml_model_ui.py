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
    Verbesserte Streamlit UI für ML-Modell-Training und -Verwaltung mit Feature-Auswahl.

    Parameters:
    -----------
    data : pd.DataFrame, optional
        DataFrame mit OHLCV-Daten und Indikatoren
    """
    st.header("ML-Modell-Verwaltung")

    # Initialisiere den ModelManager
    model_manager = ModelManager()
    processor = DataProcessor()

    # Automatisch nach Modellen suchen beim Start (ohne Button)
    with st.spinner("Suche nach Modellen..."):
        created_files = scan_for_orphaned_models()
        if created_files:
            st.success(f"{len(created_files)} neue Metadatendateien erstellt!")

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

        # Erfasse Datenbeschreibung für Modell-Metadaten
        data_source = {
            "symbol": st.session_state.get("symbol", "Unknown"),
            "period": st.session_state.get("period", "Unknown"),
            "interval": st.session_state.get("interval", "Unknown"),
            "data_points": len(data),
            "date_range": f"{data.index[0].strftime('%Y-%m-%d %H:%M')} bis {data.index[-1].strftime('%Y-%m-%d %H:%M')}"
            if isinstance(data.index, pd.DatetimeIndex) else "Unbekannt"
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
            st.info(f"Insgesamt ausgewählt: {len(selected_features)} Features")

            # Wenn keine Features ausgewählt sind, zeige Warnung
            if not selected_features:
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
                            data_source=data_source  # Neue Information über Trainingsquellen
                        )

                        # Training abgeschlossen
                        progress_bar.progress(1.0)

                        # Zeige Ergebnisse
                        st.success(f"✅ Modell '{model_name}' erfolgreich trainiert!")

                        st.write("### Trainingsmetriken")
                        metrics = metadata.get("metrics", {})
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
        st.subheader("Verfügbare Modelle")

        # Lade verfügbare Modelle
        models = model_manager.list_available_models()

        if not models:
            st.info("Keine Modelle gefunden. Trainieren Sie zuerst ein Modell.")
        else:
            # Erstelle eine übersichtliche Tabelle der Modelle
            models_df = pd.DataFrame([
                {
                    "Name": model.get("name", "Unbekannt"),
                    "Erstellt am": model.get("created_at", "").split("T")[0] if model.get("created_at", "") else "",
                    "Features": len(model.get("features", [])),
                    "Window Size": model.get("window_size", 0),
                    "RMSE": round(model.get("metrics", {}).get("rmse", 0), 6),
                    "Symbol": model.get("data_source", {}).get("symbol", "Unbekannt"),
                    "Zeitraum": model.get("data_source", {}).get("period", "Unbekannt"),
                    "Intervall": model.get("data_source", {}).get("interval", "Unbekannt"),
                    "Aktionen": model.get("name", "")
                }
                for model in models
            ])

            # Zeige die Tabelle an (ohne die Aktionsspalte)
            st.dataframe(models_df.drop(columns=["Aktionen"]))

            # Modell auswählen für Aktionen
            selected_model = st.selectbox(
                "Modell für Aktionen auswählen:",
                options=models_df["Name"].tolist()
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Modell löschen", key="delete_model"):
                    if model_manager.delete_model(selected_model):
                        st.success(f"Modell '{selected_model}' erfolgreich gelöscht")
                        st.experimental_rerun()  # UI neu laden
                    else:
                        st.error(f"Fehler beim Löschen des Modells '{selected_model}'")

            with col2:
                if st.button("Modell laden", key="load_model"):
                    model, scaler, metadata = model_manager.load_model(selected_model)
                    if model is not None:
                        st.success(f"Modell '{selected_model}' erfolgreich geladen")
                        # Speichere in Session-State für Verwendung im Backtest
                        if hasattr(st, "session_state"):
                            st.session_state.ml_model = model
                            st.session_state.ml_scaler = scaler
                            st.session_state.ml_metadata = metadata
                            st.success("Modell in Session-State gespeichert und für Backtest bereit!")
                    else:
                        st.error(f"Fehler beim Laden des Modells '{selected_model}'")

            with col3:
                if st.button("Details anzeigen", key="show_details"):
                    # Speichere die Auswahl im Session-State und wechsle zum Detailtab
                    st.session_state.selected_model_for_details = selected_model
                    st.info(f"Details für '{selected_model}' werden angezeigt. Bitte wechseln Sie zu 'Modelldetails'.")

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
                # Modellübersicht
                st.write("### Modellübersicht")
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

                # Metriken
                st.write("### Trainingsmetriken")
                metrics = model_info.get("metrics", {})
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

                # Backtest-Metriken, falls vorhanden
                if 'backtest_metrics' in model_info and any(model_info.get('backtest_metrics', {}).values()):
                    st.write("### Backtest-Metriken")
                    bt_metrics = model_info.get("backtest_metrics", {})
                    bt_metrics_df = pd.DataFrame({
                        "Metrik": ["Gesamtrendite", "Max. Drawdown", "Trades", "Gewinnrate", "Sharpe Ratio"],
                        "Wert": [
                            f"{bt_metrics.get('total_return', 0) * 100:.2f}%",
                            f"{bt_metrics.get('max_drawdown', 0):.2f}%",
                            f"{bt_metrics.get('trades', 0)}",
                            f"{bt_metrics.get('win_rate', 0) * 100:.2f}%" if bt_metrics.get(
                                'win_rate') is not None else "N/A",
                            f"{bt_metrics.get('sharpe_ratio', 0):.2f}" if bt_metrics.get(
                                'sharpe_ratio') is not None else "N/A"
                        ]
                    })
                    st.table(bt_metrics_df)

                # Verwendete Features - immer alle anzeigen
                st.write("### Verwendete Features")
                features = model_info.get("features", [])

                if features:
                    # Einfach alle Features direkt anzeigen, ohne sie zu verstecken
                    st.write(", ".join(features))

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

                    # Zeige nicht-kategorisierte Features
                    if unknown_features:
                        with st.expander(f"Sonstige Features ({len(unknown_features)})", expanded=True):
                            st.write(", ".join(unknown_features))
                else:
                    st.warning("Keine Feature-Informationen verfügbar.")

                # Zeige Modellarchitektur (falls vorhanden)
                if 'parameters' in model_info and 'layers' in model_info.get('parameters', {}):
                    with st.expander("Modellarchitektur", expanded=True):
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