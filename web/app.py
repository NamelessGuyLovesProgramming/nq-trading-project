import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

from src.custom_html_writer import write_html_with_custom_interaction
from web.custom_chart_component import create_candlestick_chart, create_backtest_chart
import plotly.graph_objects as go
from datetime import datetime
import importlib
import glob

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
        ["Daten laden", "ML-Modell trainieren", "Backtest durchführen", "Visualisieren"]
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

                # Training mit Progress-Bar
                history = model.train(
                    X_train, y_train,
                    X_test, y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[StreamlitCallback()]
                )

                # Zeige Trainingshistorie
                st.subheader("Trainingshistorie")
                hist_df = pd.DataFrame(history.history)
                st.line_chart(hist_df)

                # Vorhersagen auf Testdaten
                predictions = model.predict(X_test)

                # Leistungsmetriken berechnen
                mse = np.mean((predictions - y_test) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - y_test))
                mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

                # Zeige Metriken
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("MSE", f"{mse:.6f}")
                    st.metric("RMSE", f"{rmse:.6f}")
                with metrics_col2:
                    st.metric("MAE", f"{mae:.6f}")
                    st.metric("MAPE", f"{mape:.2f}%")

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
        create_candlestick_chart(st.session_state.data, indicators=selected_indicators)

        # Download-Button für Chart
        if st.button("Chart als HTML herunterladen"):
            # Verwende die bestehende ChartVisualizer-Klasse
            visualizer = ChartVisualizer()
            chart = visualizer.plot_candlestick_with_indicators(
                st.session_state.data,
                indicators=selected_indicators
            )

            # Erstelle temporäre Datei
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as f:
                temp_path = f.name
                write_html_with_custom_interaction(chart, temp_path)

            # Lese die Datei und biete Download an
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
        create_candlestick_chart(
            st.session_state.data,
            indicators=selected_indicators,
            signals=st.session_state.signals
        )

    elif chart_type == "Backtest-Ergebnisse":
        if st.session_state.backtest_results is None:
            st.warning("Keine Backtest-Ergebnisse vorhanden. Bitte führen Sie zuerst einen Backtest durch.")
            return

        # Erstelle und rendere Backtest-Chart mit benutzerdefinierten Interaktionen
        create_backtest_chart(st.session_state.backtest_results)

# Hauptfunktion
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


# App ausführen
if __name__ == "__main__":
    main()