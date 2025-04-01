import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf

from src.custom_html_writer import write_html_with_custom_interaction
from web.custom_chart_component import create_candlestick_chart, create_backtest_chart, downsample_data, \
    filter_weekend_days
import plotly.graph_objects as go
from datetime import datetime
import importlib
import glob
from web.ml_model_ui import ml_model_ui

# Füge Root-Verzeichnis zum Pfad hinzu für richtige Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiere die benötigten Module aus dem bestehenden Projekt
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.models.lstm import LSTMModel
from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import calculate_performance_metrics
from src.visualization.charts import ChartVisualizer

# Importiere alle verfügbaren Strategien
from src.strategies.ml_strategy import MLStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.combined import CombinedStrategy
from src.strategies.volume_profile import VolumeProfileStrategy
from src.strategies.market_regime import MarketRegimeStrategy
from src.strategies.ensemble import EnsembleStrategy
from src.strategies.risk_managed import RiskManagedStrategy

# Konstanten aus config
from config import (
    DEFAULT_SYMBOL, DEFAULT_INTERVAL, DEFAULT_PERIOD,
    RSI_OVERBOUGHT, RSI_OVERSOLD, VOLUME_THRESHOLD,
    INITIAL_CAPITAL, COMMISSION, WINDOW_SIZE, EPOCHS, BATCH_SIZE
)

# Konfiguration der Streamlit-App
st.set_page_config(
    page_title="NQ-Trading-Backtest-Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App-Titel und Beschreibung
st.title("NQ-Trading-Backtest-Tool")
st.markdown("""
    Ein leistungsstarkes Tool zum Backtesting von Handelsstrategien für den Nasdaq-100 Future und andere Märkte.
    Diese Web-UI vereinfacht die Nutzung des Kommandozeilen-Tools.
""")

# Initialisierung der Session-State-Variablen
if 'data' not in st.session_state:
    st.session_state.data = None
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'signals' not in st.session_state:
    st.session_state.signals = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Seitenleiste für Hauptparameter
with st.sidebar:
    st.header("Hauptaktionen")
    action = st.radio(
        "Wählen Sie eine Aktion:",
        ["Daten laden", "ML-Modell trainieren", "Backtest durchführen", "Visualisieren", "ML-Modell-Verwaltung"]  # Neuer Menüpunkt
    )

    st.header("Datenparameter")
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
    period = st.selectbox(
        "Zeitraum",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=2  # Default ist 1mo
    )
    interval = st.selectbox(
        "Intervall",
        ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        index=0  # Default ist 1m
    )

    # Warnung für unvereinbare Kombinationen
    if interval == "1m" and period not in ["1d", "5d"]:
        st.warning("⚠️ 1m-Intervall ist nur für Zeiträume bis zu 7 Tagen verfügbar.")

    custom_file = st.text_input("Benutzerdefinierte Datei (optional)", value="")
    combine_all_years = st.checkbox("Alle nq-1m* Dateien kombinieren")


# Daten laden
def load_data():
    st.header("Daten laden")

    # Füge einen Knopf hinzu, um den Ladevorgang zu starten
    if st.button("Daten laden starten"):
        with st.spinner("Daten werden geladen..."):
            try:
                # Erstelle DataFetcher-Instanz
                fetcher = DataFetcher(symbol=symbol)

                # Lade Daten basierend auf den Parametern
                if combine_all_years:
                    data = fetcher.load_custom_file("nq-1m*.csv")
                    st.success("Alle nq-1m* Dateien wurden kombiniert und geladen.")
                elif custom_file:
                    data = fetcher.fetch_data(period=period, interval=interval, force_download=True,
                                              custom_file=custom_file)
                    st.success(f"Benutzerdefinierte Datei '{custom_file}' wurde geladen.")
                elif period.startswith("nq-1m"):
                    data = fetcher.load_nq_minute_data(period)
                    st.success(f"NQ 1-Minuten-Daten für Periode: {period} wurden geladen.")
                else:
                    data = fetcher.fetch_data(period=period, interval=interval, force_download=True)
                    st.success(f"Daten für {symbol} mit Periode {period} und Intervall {interval} wurden geladen.")

                # Fehlerprüfung
                if data is None or data.empty:
                    st.error("Keine Daten konnten geladen werden. Überprüfen Sie die Parameter.")
                    return None

                # Zeige Datenübersicht
                st.write(f"Datenzeitraum: {data.index[0]} bis {data.index[-1]}")
                st.write(f"Anzahl der Datenpunkte: {len(data)}")

                # Füge technische Indikatoren hinzu
                processor = DataProcessor()
                data_with_indicators = processor.add_technical_indicators(data)

                # Zeige die ersten Zeilen
                st.subheader("Datenübersicht (erste 5 Zeilen)")
                st.dataframe(data_with_indicators.head())

                # Speichere in Session-State
                st.session_state.data = data_with_indicators

                return data_with_indicators

            except Exception as e:
                st.error(f"Fehler beim Laden der Daten: {e}")
                st.exception(e)
                return None
    else:
        # Zeige einen Hinweis, wenn noch keine Daten geladen wurden
        if st.session_state.data is None:
            st.info("Klicke auf 'Daten laden starten', um den Ladevorgang zu beginnen.")
        else:
            # Wenn bereits Daten geladen wurden, zeige eine Zusammenfassung
            st.success("Daten wurden bereits geladen.")
            st.write(f"Datenzeitraum: {st.session_state.data.index[0]} bis {st.session_state.data.index[-1]}")
            st.write(f"Anzahl der Datenpunkte: {len(st.session_state.data)}")

            # Option zum Neuladen anbieten
            st.write("Drücke den 'Daten laden starten' Button, um neue Daten mit den aktuellen Parametern zu laden.")

# ML-Modell trainieren
def train_model():
    st.header("ML-Modell trainieren")

    # Parameter für Modelltraining
    col1, col2 = st.columns(2)
    with col1:
        window_size = st.slider("Fenstergröße", min_value=10, max_value=120, value=WINDOW_SIZE, step=5)
        epochs = st.slider("Anzahl der Trainingsepochen", min_value=10, max_value=200, value=EPOCHS, step=10)
    with col2:
        batch_size = st.slider("Batch-Größe", min_value=8, max_value=128, value=BATCH_SIZE, step=8)
        test_size = st.slider("Testdaten-Anteil", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    # Training starten
    if st.button("Modell trainieren"):
        if st.session_state.data is None:
            st.error("Bitte laden Sie zuerst Daten.")
            return

        with st.spinner("Modell wird trainiert..."):
            try:
                # Erstelle Ausgabeverzeichnis für Modelle
                models_dir = os.path.join('output', 'models')
                os.makedirs(models_dir, exist_ok=True)

                # Daten vorbereiten
                processor = DataProcessor()
                X_train, y_train, X_test, y_test, scaler = processor.prepare_data_for_ml(
                    st.session_state.data, window_size=window_size, test_size=test_size
                )

                # Prüfe, ob genug Daten vorhanden sind
                if X_train.shape[0] < 10 or X_test.shape[0] < 5:
                    st.warning("Sehr wenige Trainings-/Testdaten. Die Modellqualität könnte leiden.")
                    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                        st.error("Keine Trainings- oder Testdaten verfügbar.")
                        return

                # Modell erstellen und trainieren
                input_shape = (window_size, X_train.shape[2])
                model = LSTMModel(input_shape, output_dir=models_dir)
                model.build_model()

                # Progress-Bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Callback für die Progress-Bar
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f} - Val Loss: {logs['val_loss']:.4f}")


                # Verwende stattdessen:
                # Zuerst Modell erstellen und kompilieren (wird bereits in model.build_model() gemacht)
                # Dann direktes Aufrufen von model.model.fit() statt model.train()
                history = model.model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    callbacks=[StreamlitCallback()],
                    verbose=1,
                    shuffle=True
                )

                # Nach dem Training, wenn du das Modell speichern möchtest:
                model_path = os.path.join('output', 'models', 'lstm_model.h5')
                model.model.save(model_path)
                print(f"Modell gespeichert unter {model_path}")
                # Zeige Trainingshistorie
                st.subheader("Trainingshistorie")
                hist_df = pd.DataFrame(history.history)
                st.line_chart(hist_df)

                # Vorhersagen auf Testdaten generieren
                predictions = model.predict(X_test)

                # Leistungsmetriken berechnen
                # Sicherere Metrikberechnung
                predictions = predictions.flatten()
                y_test = y_test.flatten()

                # Überprüfe Formen vor der Berechnung
                print(f"Vorhersagen-Form: {predictions.shape}, y_test-Form: {y_test.shape}")

                # Stelle sicher, dass die Formen übereinstimmen
                min_len = min(len(predictions), len(y_test))
                predictions = predictions[:min_len]
                y_test = y_test[:min_len]

                # Jetzt berechne die Metriken
                mse = np.mean(np.square(predictions - y_test))
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - y_test))

                # Vorsichtig bei der MAPE-Berechnung, um Division durch Null zu vermeiden
                non_zero_mask = (y_test != 0)
                if np.any(non_zero_mask):
                    mape = np.mean(
                        np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
                else:
                    mape = np.nan
                # Speichere Modell in Session-State
                st.session_state.model = model

                st.success("Modell erfolgreich trainiert und gespeichert!")

            except Exception as e:
                st.error(f"Fehler beim Training des Modells: {e}")
                st.exception(e)


# Backtest durchführen
def run_backtest():
    st.header("Backtest durchführen")

    # Überprüfe, ob Daten geladen sind
    if st.session_state.data is None:
        st.error("Bitte laden Sie zuerst Daten.")
        return

    # Strategieauswahl
    strategy_type = st.selectbox(
        "Strategie auswählen",
        ["ml", "mean_reversion", "combined", "volume", "regime", "ensemble"]
    )

    # Risikomanagement
    risk_management = st.checkbox("Risikomanagement aktivieren")

    # Container für strategie-spezifische Parameter
    strategy_params = st.expander("Strategie-Parameter", expanded=True)

    # Parameter je nach Strategie
    if strategy_type == "ml":
        if st.session_state.model is None:
            ml_model_path = st.text_input(
                "Pfad zum trainierten Modell",
                value=os.path.join("output", "models", "lstm_model.h5")
            )

            # Prüfe, ob Modell existiert
            if not os.path.exists(ml_model_path):
                st.warning(
                    f"Modell unter {ml_model_path} nicht gefunden. Bitte trainieren Sie zuerst ein Modell oder geben Sie einen gültigen Pfad an.")

            with strategy_params:
                ml_threshold = st.slider(
                    "Signal-Schwellenwert",
                    min_value=0.001,
                    max_value=0.05,
                    value=0.005,
                    step=0.001,
                    format="%.3f"
                )
                window_size = st.slider("Fenstergröße", min_value=10, max_value=120, value=WINDOW_SIZE, step=5)
        else:
            with strategy_params:
                ml_threshold = st.slider(
                    "Signal-Schwellenwert",
                    min_value=0.001,
                    max_value=0.05,
                    value=0.005,
                    step=0.001,
                    format="%.3f"
                )
                window_size = WINDOW_SIZE  # Verwende den gleichen Wert wie beim Training
    if strategy_type == "ml":
        # Überprüfe, ob ein Modell in der Session vorhanden ist
        if hasattr(st.session_state, 'ml_model') and hasattr(st.session_state, 'ml_scaler'):
            model = st.session_state.ml_model
            scaler = st.session_state.ml_scaler
            metadata = st.session_state.ml_metadata if hasattr(st.session_state, 'ml_metadata') else None

            # Verwende die Features des geladenen Modells, falls verfügbar
            selected_features = metadata.get('features') if metadata else None

            # Erstelle ML-Strategie mit dem geladenen Modell
            strategy = MLStrategy(
                model,
                scaler,
                window_size=window_size,
                threshold=ml_threshold,
                selected_features=selected_features
            )
        else:
            print()
    # Bestehender Code für den Fall, dass kein Modell geladen ist

    elif strategy_type == "mean_reversion":
        with strategy_params:
            rsi_overbought = st.slider("RSI Überkauft", min_value=50, max_value=90, value=RSI_OVERBOUGHT)
            rsi_oversold = st.slider("RSI Überverkauft", min_value=10, max_value=50, value=RSI_OVERSOLD)
            bb_trigger = st.slider("Bollinger-Band Trigger", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

    elif strategy_type == "volume":
        with strategy_params:
            volume_threshold = st.slider(
                "Volumen-Schwellenwert",
                min_value=1.0,
                max_value=5.0,
                value=VOLUME_THRESHOLD,
                step=0.1
            )
            lookback = st.slider("Lookback-Periode", min_value=5, max_value=50, value=20, step=5)

    elif strategy_type == "combined":
        with strategy_params:
            trend_weight = st.slider("Trend-Gewicht", min_value=0.0, max_value=1.0, value=0.4, step=0.1)
            momentum_weight = st.slider("Momentum-Gewicht", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            volatility_weight = st.slider("Volatilitäts-Gewicht", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

            # Normalisierung der Gewichte
            total = trend_weight + momentum_weight + volatility_weight
            if total > 0:
                trend_weight /= total
                momentum_weight /= total
                volatility_weight /= total
                st.info(
                    f"Normalisierte Gewichte: Trend={trend_weight:.2f}, Momentum={momentum_weight:.2f}, Volatilität={volatility_weight:.2f}")

    elif strategy_type == "ensemble":
        with strategy_params:
            voting_method = st.selectbox(
                "Abstimmungsmethode",
                ["majority", "unanimous", "weighted"],
                index=0
            )

    # Risikomanagement-Parameter, falls aktiviert
    if risk_management:
        risk_params = st.expander("Risikomanagement-Parameter", expanded=True)
        with risk_params:
            risk_per_trade = st.slider(
                "Risiko pro Trade (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            ) / 100.0

            position_size_method = st.selectbox(
                "Positionsgrößen-Methode",
                ["fixed", "percent", "atr"],
                index=0
            )

            if position_size_method == "atr":
                atr_risk_multiplier = st.slider(
                    "ATR Risiko-Multiplikator",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.5,
                    step=0.1
                )

            max_drawdown = -st.slider(
                "Maximaler Drawdown (%)",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            ) / 100.0

    # Backtest-Parameter
    backtest_params = st.expander("Backtest-Parameter", expanded=True)
    with backtest_params:
        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input(
                "Anfangskapital",
                min_value=1000,
                max_value=1000000,
                value=INITIAL_CAPITAL,
                step=1000
            )
        with col2:
            commission = st.number_input(
                "Provision (%)",
                min_value=0.0,
                max_value=1.0,
                value=COMMISSION * 100,
                step=0.01,
                format="%.2f"
            ) / 100.0

    # Backtest starten
    if st.button("Backtest starten"):
        with st.spinner("Backtest wird durchgeführt..."):
            try:
                # Erstelle die ausgewählte Strategie
                if strategy_type == "ml":
                    # Lade oder verwende das Modell
                    if st.session_state.model is None:
                        # Lade das Modell von der Festplatte
                        if not os.path.exists(ml_model_path):
                            st.error(f"Modell unter {ml_model_path} nicht gefunden.")
                            return

                        # Lade das Modell
                        input_shape = (window_size, len(st.session_state.data.columns))
                        model = LSTMModel(input_shape)
                        model.load_model(ml_model_path)

                        # Bereite Daten vor
                        processor = DataProcessor()
                        _, _, _, _, scaler = processor.prepare_data_for_ml(
                            st.session_state.data, window_size=window_size
                        )
                    else:
                        # Verwende das trainierte Modell aus der Session
                        model = st.session_state.model
                        processor = DataProcessor()
                        _, _, _, _, scaler = processor.prepare_data_for_ml(
                            st.session_state.data, window_size=window_size
                        )

                    # Erstelle ML-Strategie
                    strategy = MLStrategy(
                        model.model,
                        scaler,
                        window_size=window_size,
                        threshold=ml_threshold
                    )

                elif strategy_type == "mean_reversion":
                    strategy = MeanReversionStrategy(
                        rsi_overbought=rsi_overbought,
                        rsi_oversold=rsi_oversold,
                        bb_trigger=bb_trigger
                    )

                elif strategy_type == "volume":
                    strategy = VolumeProfileStrategy(
                        volume_threshold=volume_threshold,
                        lookback=lookback
                    )

                elif strategy_type == "combined":
                    weights = {
                        'trend': trend_weight,
                        'momentum': momentum_weight,
                        'volatility': volatility_weight
                    }
                    strategy = CombinedStrategy(
                        weights=weights,
                        threshold=0.15
                    )

                elif strategy_type == "regime":
                    strategy = MarketRegimeStrategy()

                elif strategy_type == "ensemble":
                    # Erstelle mehrere Strategien für das Ensemble
                    mr_strategy = MeanReversionStrategy(
                        rsi_overbought=70,
                        rsi_oversold=30
                    )
                    vol_strategy = VolumeProfileStrategy(
                        volume_threshold=1.5
                    )
                    combined_strategy = CombinedStrategy(
                        weights={'trend': 0.4, 'momentum': 0.3, 'volatility': 0.3}
                    )

                    strategy = EnsembleStrategy(
                        strategies=[mr_strategy, vol_strategy, combined_strategy],
                        voting_method=voting_method
                    )

                # Optional: Mit Risikomanagement umhüllen
                if risk_management:
                    original_strategy = strategy

                    atr_multiplier = 1.5
                    if position_size_method == "atr":
                        atr_multiplier = atr_risk_multiplier

                    strategy = RiskManagedStrategy(
                        original_strategy,
                        max_drawdown=max_drawdown,
                        volatility_filter=True,
                        risk_per_trade=risk_per_trade,
                        position_size_method=position_size_method,
                        atr_risk_multiplier=atr_multiplier
                    )

                # Backtest durchführen
                backtest_engine = BacktestEngine(
                    st.session_state.data,
                    strategy,
                    initial_capital=initial_capital,
                    commission=commission
                )
                results = backtest_engine.run()

                # Speichere Ergebnisse und Signale
                st.session_state.backtest_results = results
                st.session_state.signals = strategy.generate_signals(st.session_state.data)

                # Zeige Ergebnisse
                st.subheader("Backtest-Ergebnisse")

                # Formatiere Kennzahlen
                metrics_df = pd.DataFrame({
                    "Metrik": [
                        "Gesamtrendite",
                        "Maximaler Drawdown",
                        "Anzahl der Trades",
                        "Gewinnrate",
                        "Sharpe Ratio"
                    ],
                    "Wert": [
                        f"{(results['portfolio_value'].iloc[-1] / initial_capital - 1) * 100:.2f}%",
                        f"{results['max_drawdown']:.2f}%",
                        f"{results['trades']}",
                        f"{results['win_rate'] * 100:.2f}%" if results['win_rate'] is not None else "N/A",
                        f"{results['sharpe_ratio']:.2f}" if results['sharpe_ratio'] is not None else "N/A"
                    ]
                })

                st.table(metrics_df)

                # Zeige Chart
                fig = backtest_engine.plot_results()
                st.pyplot(fig)

                # Berechne Benchmark-Renditen (Buy & Hold)
                benchmark_returns = st.session_state.data['Close'].pct_change()

                # Berechne erweiterte Metriken mit Benchmark
                additional_metrics = calculate_performance_metrics(
                    results['returns'],
                    benchmark_returns=benchmark_returns
                )

                # Zeige erweiterte Metriken
                st.subheader("Erweiterte Performance-Metriken")

                # Erstelle eine schönere Tabelle
                metrics_to_show = [
                    "annualized_return", "annualized_volatility",
                    "calmar_ratio", "profit_factor", "alpha", "beta"
                ]

                metrics_names = {
                    "annualized_return": "Annualisierte Rendite",
                    "annualized_volatility": "Annualisierte Volatilität",
                    "calmar_ratio": "Calmar Ratio",
                    "profit_factor": "Profit Faktor",
                    "alpha": "Alpha",
                    "beta": "Beta"
                }

                adv_metrics = []
                for metric in metrics_to_show:
                    if metric in additional_metrics and additional_metrics[metric] is not None:
                        value = additional_metrics[metric]
                        if metric == "annualized_return":
                            value = f"{value * 100:.2f}%"
                        elif metric == "annualized_volatility":
                            value = f"{value * 100:.2f}%"
                        else:
                            value = f"{value:.4f}"

                        adv_metrics.append({
                            "Metrik": metrics_names.get(metric, metric),
                            "Wert": value
                        })

                st.table(pd.DataFrame(adv_metrics))

                st.success("Backtest erfolgreich durchgeführt!")

            except Exception as e:
                st.error(f"Fehler während des Backtests: {e}")
                st.exception(e)


# Visualisierungen
# Updated visualize_data function for web/app.py

# Updated visualize_data function for web/app.py

# Aktualisierte Funktion in web/app.py, um die Option zum Überspringen von Wochenenden anzubieten

def visualize_data():
    st.header("Daten visualisieren")

    # Überprüfe, ob Daten geladen sind
    if st.session_state.data is None:
        st.error("Bitte laden Sie zuerst Daten.")
        return

    # Visualisierungsoptionen
    chart_type = st.selectbox(
        "Chart-Typ",
        ["Candlestick mit Indikatoren", "Signale anzeigen", "Backtest-Ergebnisse"]
    )

    # Sicherstellen, dass Datums-Index korrekt ist
    try:
        if not isinstance(st.session_state.data.index, pd.DatetimeIndex):
            st.warning("Daten haben keinen DateTime-Index. Versuche, Index zu konvertieren...")
            st.session_state.data.index = pd.to_datetime(st.session_state.data.index)
    except Exception as e:
        st.error(f"Fehler beim Konvertieren des Index: {e}")
        st.warning(
            "Die Visualisierung könnte Probleme haben. Bitte stellen Sie sicher, dass die Daten korrekt geladen wurden.")

    # Sicherstellen, dass OHLCV-Spalten numerisch sind
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlcv_cols:
        if col in st.session_state.data.columns and not pd.api.types.is_numeric_dtype(st.session_state.data[col]):
            try:
                st.session_state.data[col] = pd.to_numeric(st.session_state.data[col], errors='coerce')
            except Exception as e:
                st.warning(f"Konnte Spalte {col} nicht in numerischen Typ konvertieren: {e}")

    # Dataset size estimation and warning
    try:
        data_size_mb = st.session_state.data.memory_usage(deep=True).sum() / (1024 * 1024)
        data_points = len(st.session_state.data)

        if data_size_mb > 50:
            st.warning(
                f"Der Datensatz ist groß ({data_points:,} Datenpunkte, ~{data_size_mb:.1f} MB). Die Visualisierung wird optimiert, um Speicherprobleme zu vermeiden.")
    except Exception as e:
        st.warning(f"Konnte Datensatzgröße nicht bestimmen: {e}")
        data_size_mb = 999  # Setze auf hohen Wert, um Optimierungen zu aktivieren
        data_points = len(st.session_state.data)

    # Chart-Optionen
    chart_options = st.expander("Chart-Optionen", expanded=True)
    with chart_options:
        col1, col2 = st.columns(2)
        with col1:
            # Neue Option: Wochenendtage überspringen
            skip_weekends = st.checkbox("Wochenendtage überspringen", value=True,
                                        help="Blendet Wochenendtage und Tage ohne Handelsdaten aus")
        with col2:
            max_points = st.slider("Maximale Anzahl von Datenpunkten",
                                   min_value=1000,
                                   max_value=20000,
                                   value=10000,
                                   step=1000,
                                   help="Reduziere diese Zahl bei Speicherproblemen.")

    # Add date range selector for large datasets
    use_date_filter = st.checkbox("Zeitraum einschränken", value=data_size_mb > 100)
    date_range = None

    if use_date_filter:
        try:
            # Create date range selector
            min_date = st.session_state.data.index.min()
            max_date = st.session_state.data.index.max()

            # Default to last month for large datasets
            default_start = max_date - pd.Timedelta(days=30) if data_size_mb > 100 else min_date

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Von", value=default_start, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("Bis", value=max_date, min_value=min_date, max_value=max_date)

            # Convert to datetime for filtering
            start_datetime = pd.Timestamp(start_date)
            end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(
                seconds=1)  # End of the selected day

            date_range = (start_datetime, end_datetime)

            # Show how many data points are in the selected range
            filtered_data = st.session_state.data[(st.session_state.data.index >= start_datetime) &
                                                  (st.session_state.data.index <= end_datetime)]
            st.info(f"Ausgewählter Zeitraum enthält {len(filtered_data):,} von {data_points:,} Datenpunkten.")
        except Exception as e:
            st.error(f"Fehler bei der Datumsbereichsauswahl: {e}")
            st.info("Versuche den Chart ohne Datumsfilter zu erstellen...")
            date_range = None

    if chart_type == "Candlestick mit Indikatoren":
        # Wähle Indikatoren
        available_indicators = [col for col in st.session_state.data.columns
                                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

        selected_indicators = st.multiselect(
            "Indikatoren auswählen",
            available_indicators,
            default=['SMA_20', 'EMA_9'] if 'SMA_20' in available_indicators and 'EMA_9' in available_indicators else []
        )

        # Erstelle und rendere Chart mit benutzerdefinierten Interaktionen
        # HIER die skip_weekends Option übergeben
        create_candlestick_chart(
            st.session_state.data,
            indicators=selected_indicators,
            date_range=date_range,
            max_points=max_points,
            skip_weekends=skip_weekends  # Neue Option übergeben
        )

        # Download-Button for chart
        if st.button("Chart als HTML herunterladen"):
            # Use the existing ChartVisualizer class
            visualizer = ChartVisualizer()

            # Filter data if date range is provided
            data_to_viz = st.session_state.data
            if date_range:
                start_date, end_date = date_range
                data_to_viz = data_to_viz[(data_to_viz.index >= start_date) &
                                          (data_to_viz.index <= end_date)]

            # Filter weekend days if requested
            if skip_weekends:
                data_to_viz = filter_weekend_days(data_to_viz)

            # Downsample if needed
            if len(data_to_viz) > max_points:
                data_to_viz = downsample_data(data_to_viz, max_points)

            chart = visualizer.plot_candlestick_with_indicators(
                data_to_viz,
                indicators=selected_indicators
            )

            # Configure chart to skip weekends if requested
            if skip_weekends:
                chart.update_xaxes(
                    rangebreaks=[dict(pattern='day of week', bounds=[5, 7])]
                )

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as f:
                temp_path = f.name
                write_html_with_custom_interaction(chart, temp_path)

            # Read file and offer download
            with open(temp_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            os.unlink(temp_path)

            st.download_button(
                label="Chart als HTML speichern",
                data=html_content,
                file_name="chart.html",
                mime="text/html"
            )

    elif chart_type == "Signale anzeigen":
        if st.session_state.signals is None:
            st.warning("Keine Signale vorhanden. Bitte führen Sie zuerst einen Backtest durch.")
            return

        # Wähle Indikatoren
        available_indicators = [col for col in st.session_state.data.columns
                                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]

        selected_indicators = st.multiselect(
            "Indikatoren auswählen",
            available_indicators,
            default=['RSI', 'MACD'] if 'RSI' in available_indicators and 'MACD' in available_indicators else []
        )

        # Erstelle und rendere Chart mit benutzerdefinierten Interaktionen
        # HIER auch die skip_weekends Option übergeben
        create_candlestick_chart(
            st.session_state.data,
            indicators=selected_indicators,
            signals=st.session_state.signals,
            date_range=date_range,
            max_points=max_points,
            skip_weekends=skip_weekends  # Neue Option übergeben
        )

    elif chart_type == "Backtest-Ergebnisse":
        if st.session_state.backtest_results is None:
            st.warning("Keine Backtest-Ergebnisse vorhanden. Bitte führen Sie zuerst einen Backtest durch.")
            return

        # Downsampling options for very large backtest results
        result_length = len(st.session_state.backtest_results.get('portfolio_value', []))

        if result_length > 5000:
            max_points = st.slider("Maximale Anzahl von Datenpunkten",
                                   min_value=1000,
                                   max_value=10000,
                                   value=5000,
                                   step=1000,
                                   help="Reduziere diese Zahl bei Speicherproblemen.")

        # Erstelle und rendere Backtest-Chart mit benutzerdefinierten Interaktionen
        create_backtest_chart(
            st.session_state.backtest_results,
            max_points=max_points
        )

def main():
    # Aktionen basierend auf ausgewähltem Tab
    if action == "Daten laden":
        load_data()
    elif action == "ML-Modell trainieren":
        train_model()
    elif action == "Backtest durchführen":
        run_backtest()
    elif action == "Visualisieren":
        visualize_data()
    elif action == "ML-Modell-Verwaltung":  # Neue Bedingung
        # Übergebe die geladenen Daten an die UI
        ml_model_ui(st.session_state.data if 'data' in st.session_state else None)


# App ausführen
if __name__ == "__main__":
    main()