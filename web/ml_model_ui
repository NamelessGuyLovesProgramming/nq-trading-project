import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from src.data.processor import DataProcessor
from src.models.model_manager import ModelManager

def ml_model_ui(data=None):
    """
    Streamlit UI für ML-Modell-Training und -Verwaltung mit Feature-Auswahl.

    Parameters:
    -----------
    data : pd.DataFrame, optional
        DataFrame mit OHLCV-Daten und Indikatoren
    """
    st.header("ML-Modell Training und Verwaltung")

    # Initialisiere den ModelManager
    model_manager = ModelManager()
    processor = DataProcessor()

    # Prüfe, ob Daten geladen sind
    if data is None:
        st.warning("⚠️ Bitte laden Sie zuerst Daten.")
        return

    # Zeige verfügbare Features an
    feature_categories = processor.get_available_features()

    # Tabs für verschiedene Modell-Funktionen
    tab1, tab2, tab3 = st.tabs(["Modell trainieren", "Modelle verwalten", "Modelldetails"])

    # Tab 1: Modell trainieren
    with tab1:
        st.subheader("Neues Modell trainieren")

        # Modellname
        model_name = st.text_input("Modellname", value=f"Model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Feature-Auswahl mit Kategorien
        st.write("### Feature-Auswahl")

        # Ein Expander für jede Feature-Kategorie
        selected_features = []

        for category, features in feature_categories.items():
            with st.expander(f"{category} ({len(features)} Features)", expanded=True if category == "Basisdaten" else False):
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
                    default_selections = ["SMA_20", "EMA_9"] if all(f in available_features for f in ["SMA_20", "EMA_9"]) else []
                elif category == "Momentum":
                    default_selections = ["RSI", "MACD"] if all(f in available_features for f in ["RSI", "MACD"]) else []
                elif category == "Volatilität":
                    default_selections = ["BB_Middle", "BB_Upper", "BB_Lower"] if all(f in available_features for f in ["BB_Middle", "BB_Upper", "BB_Lower"]) else []

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
                        test_size=test_size
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
                        "testing_samples": metadata.get("testing_samples", 0)
                    })

                except Exception as e:
                    st.error(f"❌ Fehler beim Training: {str(e)}")
                    st.exception(e)

    # Tab 2: Modelle verwalten
    with tab2:
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
                    "Erstellt am": model.get("created_at", "").split("T")[0],
                    "Features": len(model.get("features", [])),
                    "Window Size": model.get("window_size", 0),
                    "RMSE": round(model.get("metrics", {}).get("rmse", 0), 6),
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

            col1, col2 = st.columns(2)

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
                        # Hier könnten wir das Modell in eine Session-Variable speichern
                        if hasattr(st, "session_state"):
                            st.session_state.ml_model = model
                            st.session_state.ml_scaler = scaler
                            st.session_state.ml_metadata = metadata
                    else:
                        st.error(f"Fehler beim Laden des Modells '{selected_model}'")

    # Tab 3: Modelldetails
    with tab3:
        st.subheader("Modelldetails")

        # Lade verfügbare Modelle für Dropdown
        models = model_manager.list_available_models()
        model_names = [model.get("name", "Unbekannt") for model in models]

        if not model_names:
            st.info("Keine Modelle gefunden. Trainieren Sie zuerst ein Modell.")
        else:
            # Modell auswählen
            detail_model = st.selectbox(
                "Modell auswählen:",
                options=model_names
            )

            # Lade detaillierte Infos
            model_info = model_manager.get_model_info(detail_model)

            if model_info:
                # Modellübersicht
                st.write("### Modellübersicht")
                st.json({
                    "name": model_info.get("name", ""),
                    "created_at": model_info.get("created_at", ""),
                    "window_size": model_info.get("window_size", 0),
                    "epochs": model_info.get("epochs", 0),
                    "batch_size": model_info.get("batch_size", 0),
                    "input_shape": model_info.get("input_shape", []),
                    "training_samples": model_info.get("training_samples", 0),
                    "testing_samples": model_info.get("testing_samples", 0)
                })

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

                # Verwendete Features
                st.write("### Verwendete Features")
                features = model_info.get("features", [])

                # Gruppiere Features nach Kategorien
                feature_by_category = {}
                for category, cat_features in feature_categories.items():
                    feature_by_category[category] = [f for f in features if f in cat_features]

                # Zeige Features nach Kategorien
                for category, cat_features in feature_by_category.items():
                    if cat_features:
                        with st.expander(f"{category} ({len(cat_features)} Features)"):
                            st.write(", ".join(cat_features))

                # Zeige Modellparameter (falls vorhanden)
                if "parameters" in model_info:
                    st.write("### Modellparameter")
                    st.json(model_info.get("parameters", {}))
            else:
                st.warning(f"Keine Details für Modell '{detail_model}' gefunden")

if __name__ == "__main__":
    # Dummy-Daten für Test
    import numpy as np

    # Erstelle Dummy-DataFrame
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1h')
    data = pd.DataFrame({
        'Open': np.random.normal(100, 5, 1000),
        'High': np.random.normal(105, 5, 1000),
        'Low': np.random.normal(95, 5, 1000),
        'Close': np.random.normal(100, 5, 1000),
        'Volume': np.random.normal(1000, 200, 1000)
    }, index=dates)

    # Füge ein paar technische Indikatoren hinzu
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['RSI'] = np.random.normal(50, 15, 1000)

    # Streamlit App
    st.set_page_config(page_title="ML-Modell UI Test", layout="wide")
    ml_model_ui(data)