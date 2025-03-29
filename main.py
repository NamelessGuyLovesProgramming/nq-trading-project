import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Projekt-Module
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.models.lstm import LSTMModel
from src.strategies import ml_strategy

from src.strategies.ml_strategy import MLStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.combined import CombinedStrategy
from src.strategies.volume_profile import VolumeProfileStrategy
from src.strategies.market_regime import MarketRegimeStrategy
from src.strategies.risk_managed import RiskManagedStrategy
from src.strategies.ensemble import EnsembleStrategy

from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import calculate_performance_metrics
from src.visualization.charts import ChartVisualizer

from src.custom_html_writer import write_html_with_custom_interaction


def main():
    # Importiere Konfigurationsparameter
    from config import (
        DEFAULT_SYMBOL, DEFAULT_INTERVAL, DEFAULT_PERIOD,
        RSI_OVERBOUGHT, RSI_OVERSOLD, VOLUME_THRESHOLD,
        INITIAL_CAPITAL, COMMISSION
    )

    parser = argparse.ArgumentParser(description='NQ Trading Backtest Tool')

    # Grundlegende Parameter
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL,
                        help=f'Symbol to download (default: {DEFAULT_SYMBOL})')
    parser.add_argument('--period', type=str, default=DEFAULT_PERIOD,
                        help=f'Data period (default: {DEFAULT_PERIOD}, options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)')
    parser.add_argument('--interval', type=str, default=DEFAULT_INTERVAL,
                        help=f'Data interval (default: {DEFAULT_INTERVAL}, options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)')

    # Neue Option für benutzerdefinierte Dateinamen
    parser.add_argument('--custom-file', type=str, default=None,
                        help='Name eines benutzerdefinierten Datensatzes im data/raw Verzeichnis (unterstützt auch Glob-Muster wie "nq-1m*.csv")')

    # Option zum Kombinieren aller verfügbaren Jahre
    parser.add_argument('--combine-all-years', action='store_true',
                        help='Kombiniert alle nq-1m* Dateien für umfassenden Backtest')

    # Aktionsmodi
    parser.add_argument('--train', action='store_true', help='Train new ML model')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--visualize', action='store_true', help='Visualize data and results')
    parser.add_argument('--repair-data', action='store_true', help='Repair/check data files')

    # Output und Logging
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='Increase verbosity (can be used multiple times)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force operations like redownloading data or training with insufficient data')

    # Strategie-Auswahl
    parser.add_argument('--strategy', type=str, default='ml',
                        choices=['ml', 'mean_reversion', 'combined', 'volume', 'regime', 'ensemble'],
                        help='Trading strategy to use')
    parser.add_argument('--risk-management', action='store_true',
                        help='Enable risk management for the strategy')
    parser.add_argument('--risk-per-trade', type=float, default=0.01,
                        help='Prozentsatz des Kapitals, das pro Trade riskiert wird (default: 0.01 = 1%%)')
    parser.add_argument('--position-size-method', type=str, default='fixed',
                        choices=['fixed', 'percent', 'atr'],
                        help='Methode zur Berechnung der Positionsgröße (default: fixed)')
    parser.add_argument('--atr-risk-multiplier', type=float, default=1.5,
                        help='Multiplikator für ATR bei ATR-basierter Positionsgrößenberechnung (default: 1.5)')
    parser.add_argument('--max-drawdown', type=float, default=-0.05,
                        help='Maximaler Drawdown, bevor die Positionsgröße reduziert wird (default: -0.05 = -5%%)')

    # Parameter für Mean-Reversion-Strategie
    parser.add_argument('--rsi-overbought', type=int, default=RSI_OVERBOUGHT,
                        help=f'RSI threshold for overbought condition (default: {RSI_OVERBOUGHT})')
    parser.add_argument('--rsi-oversold', type=int, default=RSI_OVERSOLD,
                        help=f'RSI threshold for oversold condition (default: {RSI_OVERSOLD})')

    # Parameter für Volume-Strategie
    parser.add_argument('--volume-threshold', type=float, default=VOLUME_THRESHOLD,
                        help=f'Volume threshold as multiple of average (default: {VOLUME_THRESHOLD})')

    # Parameter für Kombinierte Strategie
    parser.add_argument('--trend-weight', type=float, default=0.4,
                        help='Weight for trend signals in combined strategy (default: 0.4)')
    parser.add_argument('--momentum-weight', type=float, default=0.3,
                        help='Weight for momentum signals in combined strategy (default: 0.3)')
    parser.add_argument('--volatility-weight', type=float, default=0.3,
                        help='Weight for volatility signals in combined strategy (default: 0.3)')

    # Parameter für Ensemble-Strategie
    parser.add_argument('--voting-method', type=str, default='majority',
                        choices=['majority', 'unanimous', 'weighted'],
                        help='Voting method for ensemble strategy (default: majority)')

    # Parameter für Backtest
    parser.add_argument('--initial-capital', type=float, default=INITIAL_CAPITAL,
                        help=f'Initial capital for backtest (default: {INITIAL_CAPITAL})')
    parser.add_argument('--commission', type=float, default=COMMISSION,
                        help=f'Commission rate for trades (default: {COMMISSION})')

    # Parameter für Modelltraining
    parser.add_argument('--window-size', type=int, default=60,
                        help='Window size for LSTM model (default: 60)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')

    args = parser.parse_args()

    # Erstelle benötigte Verzeichnisse
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'charts'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)

    # Verbositätsstufe festlegen
    verbose_level = args.verbose

    # Log-Datei einrichten
    log_file = os.path.join(args.output_dir, 'execution_log.txt')
    logging_mode = 'a' if os.path.exists(log_file) else 'w'
    with open(log_file, logging_mode) as f:
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Ausführung gestartet am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Befehlszeilenargumente: {vars(args)}\n")
        f.write(f"{'=' * 50}\n")

    print(f"=== NQ Trading Backtest Tool ===")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.period}")
    print(f"Interval: {args.interval}")

    # Prüfe auf gültige Periodenwerte (aktualisiert für nq-1m*)
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
    if args.period not in valid_periods and not args.period.startswith('nq-1m'):
        print(
            f"Warnung: Unbekannter Periodenwert '{args.period}'. Gültige Werte sind: {', '.join(valid_periods)} oder nq-1m*")
        if not args.combine_all_years and not args.custom_file:
            print(f"Verwende Standardwert '1y'")
            args.period = '1y'
    else:
        if args.period.startswith('nq-1m'):
            print(f"Verwende NQ 1-Minuten-Daten für Periode: {args.period}")

    # Prüfe auf gültige Intervallwerte
    valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    if args.interval not in valid_intervals:
        print(f"Warnung: Unbekannter Intervallwert '{args.interval}'. Gültige Werte sind: {', '.join(valid_intervals)}")
        print(f"Verwende Standardwert '1d'")
        args.interval = '1d'

    # Führe Datenreparatur aus, wenn angefordert
    if args.repair_data:
        print("\nDaten reparieren...")
        try:
            from repair_csv import repair_csv_files
            repair_csv_files(data_dir='data/raw', force_download=args.force)
        except Exception as e:
            print(f"Fehler bei der Datenreparatur: {e}")
            import traceback
            traceback.print_exc()

    # Daten holen
    print("\n1. Daten laden...")
    fetcher = DataFetcher(symbol=args.symbol)

    # Debugging-Ausgabe
    print(f"Debug - Period-Parameter: {args.period}")
    print(f"Debug - Custom-File: {args.custom_file}")
    print(f"Debug - Combine-All-Years: {args.combine_all_years}")

    # Variable 'data' vor der Verwendung initialisieren
    data = None

    try:
        if args.combine_all_years:
            # Kombiniert alle nq-1m* Dateien
            print("Kombiniere alle nq-1m* Dateien für umfassenden Backtest...")
            data = fetcher.load_custom_file("nq-1m*.csv")
        elif args.custom_file:
            # Lade benutzerdefinierte Datei
            print(f"Lade benutzerdefinierte Datei: {args.custom_file}")
            data = fetcher.fetch_data(period=args.period, interval=args.interval, force_download=True,
                                      custom_file=args.custom_file)
        elif args.period.startswith("nq-1m"):
            # Nutze spezielle NQ 1-Minuten-Daten
            print(f"Lade NQ 1-Minuten-Daten für Periode: {args.period}")
            data = fetcher.load_nq_minute_data(args.period)
        else:
            # Standardverhalten
            data = fetcher.fetch_data(period=args.period, interval=args.interval, force_download=True)

        # Prüfe, ob Daten erfolgreich geladen wurden
        if data is None or data.empty:
            print("Fehler: Keine Daten konnten geladen werden. Überprüfen Sie die Datei oder Parameter.")
            return

        print(f"Daten geladen: {len(data)} Einträge")
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        import traceback
        traceback.print_exc()
        print("Bitte überprüfen Sie, ob die Dateien korrekt im 'data/raw' Verzeichnis vorhanden sind.")
        return

    # Daten verarbeiten
    print("\n2. Daten verarbeiten...")
    processor = DataProcessor()

    # Überprüfe, ob die Daten korrekt geladen wurden
    print(f"Rohdaten Shape: {data.shape}")
    print(f"Erster Eintrag: {data.head(1)}")

    # Füge technische Indikatoren hinzu mit zusätzlicher Fehlerbehandlung
    try:
        data_with_indicators = processor.add_technical_indicators(data)
        print(
            f"Technische Indikatoren hinzugefügt, neue Spalten: {set(data_with_indicators.columns) - set(data.columns)}")
    except Exception as e:
        print(f"Fehler beim Hinzufügen von Indikatoren: {e}")
        print("Fahre mit Originaldaten fort...")
        data_with_indicators = data.copy()

    # Visualisierung
    if args.visualize:
        print("\n3. Daten visualisieren...")
        visualizer = ChartVisualizer()
        chart = visualizer.plot_candlestick_with_indicators(
            data_with_indicators,
            indicators=['SMA_20', 'EMA_9', 'BB_Upper', 'BB_Lower']
        )
        # Auch bei der Visualisierung mit Signalen:
        chart_path = os.path.join(args.output_dir, 'charts', f'chart_with_signals_{args.strategy}.html')
        write_html_with_custom_interaction(chart, chart_path)
        print(f"Chart mit Signalen und interaktiven Funktionen erstellt und gespeichert unter {chart_path}")

    # ML-Modell Training
    if args.train:
        print("\n4. ML-Modell trainieren...")

        try:
            # Importiere config für Standardparameter
            from config import WINDOW_SIZE, BATCH_SIZE, EPOCHS

            window_size = WINDOW_SIZE
            print(f"Verwende Fenstergrößoe (window_size): {window_size}")

            # Erstelle Ausgabeverzeichnis für Modelle
            models_dir = os.path.join(args.output_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)

            # Prüfe, ob die Datenmenge ausreichend ist
            min_required_data = window_size * 3  # Mind. 3 Sequenzen
            if len(data_with_indicators) < min_required_data:
                print(
                    f"Warnung: Zu wenig Daten für Modelltraining (benötigt: {min_required_data}, vorhanden: {len(data_with_indicators)})")
                print("Erwäge, einen längeren Zeitraum mit --period zu wählen")
                if not args.force:
                    print("Training wird übersprungen. Mit --force Flag erzwingen.")
                    return

            # Daten vorbereiten mit verbesserter Fehlerbehandlung
            try:
                X_train, y_train, X_test, y_test, scaler = processor.prepare_data_for_ml(
                    data_with_indicators, window_size=window_size
                )

                print(f"Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")

                # Prüfe, ob genug Daten vorhanden sind
                if X_train.shape[0] < 10 or X_test.shape[0] < 5:
                    print("Warnung: Sehr wenige Trainings-/Testdaten. Die Modellqualität könnte leiden.")
                    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                        raise ValueError("Keine Trainings- oder Testdaten verfügbar.")

            except Exception as e:
                print(f"Fehler bei der Datenvorbereitung: {e}")
                print("Versuche alternative Datenvorbereitung mit kürzerem Fenster...")

                # Versuche mit kürzerem Fenster
                window_size = min(30, len(data_with_indicators) // 4)
                X_train, y_train, X_test, y_test, scaler = processor.prepare_data_for_ml(
                    data_with_indicators, window_size=window_size
                )
                print(f"Alternative Datenvorbereitung: Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")

            # Modell erstellen und trainieren
            input_shape = (window_size, X_train.shape[2])
            model = LSTMModel(input_shape, output_dir=models_dir)

            # Baue und trainiere das Modell mit Fehlerbehandlung
            try:
                model.build_model()

                # Passe Batch-Größe an Datenmenge an
                adjusted_batch_size = min(BATCH_SIZE, max(1, len(X_train) // 10))

                history = model.train(
                    X_train, y_train,
                    X_test, y_test,
                    epochs=EPOCHS,
                    batch_size=adjusted_batch_size
                )

                # Trainingshistorie plotten
                model.plot_training_history(history)
                print("Modell trainiert und gespeichert")

                # Vorhersagen auf Testdaten
                predictions = model.predict(X_test)

                # Umkehrtransformation (falls nötig)
                y_test_inv = y_test
                predictions_inv = predictions

                # Leistungsmetriken berechnen
                mse = np.mean((predictions_inv - y_test_inv) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions_inv - y_test_inv))

                # Berechne Durchschnittlichen prozentualen Fehler
                mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100

                print("\nModell-Leistungsmetriken:")
                print(f"MSE: {mse:.6f}")
                print(f"RMSE: {rmse:.6f}")
                print(f"MAE: {mae:.6f}")
                print(f"MAPE: {mape:.2f}%")

                # Speichere Metriken in einer Datei
                metrics_file = os.path.join(models_dir, 'model_metrics.txt')
                with open(metrics_file, 'w') as f:
                    f.write(f"Trainingsdatum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Symbol: {args.symbol}\n")
                    f.write(f"Zeitraum: {args.period}\n")
                    f.write(f"Intervall: {args.interval}\n")
                    f.write(f"Window Size: {window_size}\n")
                    f.write(f"Trainingsdata Shape: {X_train.shape}\n")
                    f.write(f"Testdata Shape: {X_test.shape}\n")
                    f.write(f"MSE: {mse:.6f}\n")
                    f.write(f"RMSE: {rmse:.6f}\n")
                    f.write(f"MAE: {mae:.6f}\n")
                    f.write(f"MAPE: {mape:.2f}%\n")

                print(f"Metriken in {metrics_file} gespeichert")

            except Exception as e:
                print(f"Fehler beim Modelltraining: {e}")
                print("Das Training konnte nicht erfolgreich abgeschlossen werden.")

        except Exception as e:
            print(f"Unerwarteter Fehler im ML-Trainingsablauf: {e}")
            import traceback
            traceback.print_exc()

    # Im Backtest-Abschnitt:
    if args.backtest:
        print("\n5. Backtesting durchführen...")

        try:
            # Strategie basierend auf Benutzerauswahl erstellen
            if args.strategy == 'ml':
                # ML-Strategie mit verbesserter Fehlerbehandlung
                model_path = os.path.join(args.output_dir, 'models', 'lstm_model.h5')
                alt_model_path = os.path.join(args.output_dir, 'models', 'simple_lstm_model.h5')
                best_model_path = os.path.join(args.output_dir, 'models', 'best_model.h5')

                # Suche nach verfügbaren Modellen
                if os.path.exists(model_path):
                    selected_model_path = model_path
                    print(f"Verwende trainiertes Modell: {model_path}")
                elif os.path.exists(best_model_path):
                    selected_model_path = best_model_path
                    print(f"Verwende bestes trainiertes Modell: {best_model_path}")
                elif os.path.exists(alt_model_path):
                    selected_model_path = alt_model_path
                    print(f"Verwende alternatives Modell: {alt_model_path}")
                else:
                    print(f"Kein trainiertes Modell gefunden. Bitte zuerst trainieren mit --train.")
                    if not args.force:
                        print("Abbruch. Verwende --force um trotzdem fortzufahren mit einer anderen Strategie.")
                        return
                    else:
                        print("--force Flag erkannt. Verwende Mean-Reversion-Strategie als Fallback.")
                        args.strategy = 'mean_reversion'

            if args.strategy == 'ml':  # Falls wir nicht zu mean_reversion gewechselt haben
                window_size = args.window_size

                try:
                    # Skalierung vorbereiten
                    _, _, _, _, scaler = processor.prepare_data_for_ml(
                        data_with_indicators, window_size=window_size
                    )

                    # Modell laden
                    feature_count = len([col for col in data_with_indicators.columns
                                         if col in ['Open', 'High', 'Low', 'Close', 'Volume',
                                                    'SMA_20', 'EMA_9', 'RSI', 'MACD', 'MACD_Hist',
                                                    'BB_Upper', 'BB_Lower', 'STOCH_k', 'ATR']])

                    input_shape = (window_size, feature_count)
                    print(f"Modell-Input-Shape: {input_shape}")

                    model = LSTMModel(input_shape)
                    model.load_model(selected_model_path)

                    strategy = MLStrategy(model.model, scaler, window_size=window_size, threshold=0.005)
                    print("ML-Strategie erfolgreich initialisiert")

                except Exception as e:
                    print(f"Fehler beim Laden des ML-Modells: {e}")
                    print("Verwende Mean-Reversion-Strategie als Fallback.")
                    args.strategy = 'mean_reversion'

            # Fallback oder direkt gewählte Strategien
            if args.strategy == 'mean_reversion':
                strategy = MeanReversionStrategy(
                    rsi_overbought=args.rsi_overbought,
                    rsi_oversold=args.rsi_oversold,
                    bb_trigger=0.7
                )
                print(
                    f"Mean-Reversion-Strategie initialisiert mit RSI-Grenzen {args.rsi_oversold}/{args.rsi_overbought}")

            elif args.strategy == 'combined':
                weights = {
                    'trend': args.trend_weight,
                    'momentum': args.momentum_weight,
                    'volatility': args.volatility_weight
                }
                strategy = CombinedStrategy(
                    weights=weights,
                    threshold=0.15
                )
                print(f"Kombinierte Strategie initialisiert mit Gewichten: {weights}")

            elif args.strategy == 'volume':
                strategy = VolumeProfileStrategy(
                    volume_threshold=args.volume_threshold,
                    lookback=20
                )
                print(f"Volumen-Strategie initialisiert mit Schwellenwert {args.volume_threshold}")

            elif args.strategy == 'regime':
                strategy = MarketRegimeStrategy()
                print("Marktregime-Strategie initialisiert")

            elif args.strategy == 'ensemble':
                # Erstelle mehrere Strategien und kombiniere sie
                mr_strategy = MeanReversionStrategy(
                    rsi_overbought=args.rsi_overbought,
                    rsi_oversold=args.rsi_oversold
                )
                vol_strategy = VolumeProfileStrategy(
                    volume_threshold=args.volume_threshold
                )
                weights = {
                    'trend': args.trend_weight,
                    'momentum': args.momentum_weight,
                    'volatility': args.volatility_weight
                }
                comb_strategy = CombinedStrategy(weights=weights)

                # Erstelle Ensemble mit den definierten Strategien
                strategy = EnsembleStrategy(
                    strategies=[mr_strategy, vol_strategy, comb_strategy],
                    voting_method=args.voting_method
                )
                print(f"Ensemble-Strategie initialisiert mit Voting-Methode '{args.voting_method}'")

            # Optional: Mit Risikomanagement umhüllen
            if args.risk_management:
                original_strategy = strategy
                strategy = RiskManagedStrategy(
                    original_strategy,
                    max_drawdown=args.max_drawdown,
                    volatility_filter=True,
                    risk_per_trade=args.risk_per_trade,
                    position_size_method=args.position_size_method,
                    atr_risk_multiplier=args.atr_risk_multiplier
                )
                print(f"Risikomanagement aktiviert für {original_strategy.name}")

            # Backtest durchführen
            print("\nStarte Backtest...")
            backtest_engine = BacktestEngine(
                data_with_indicators,
                strategy,
                initial_capital=args.initial_capital,
                commission=args.commission
            )
            results = backtest_engine.run()

            # Erweiterte Metriken berechnen und anzeigen
            if 'returns' in results:
                # Berechne Benchmark-Renditen (Buy & Hold)
                benchmark_returns = data_with_indicators['Close'].pct_change()

                # Berechne erweiterte Metriken mit Benchmark
                additional_metrics = calculate_performance_metrics(
                    results['returns'],
                    benchmark_returns=benchmark_returns
                )
                print("\n=== Erweiterte Performance-Metriken ===")
                for metric, value in additional_metrics.items():
                    if isinstance(value, float):
                        print(f"{metric}: {value:.4f}")

                # Speichere Metriken in Datei
                metrics_file = os.path.join(args.output_dir, 'reports', f'backtest_metrics_{args.strategy}.txt')
                with open(metrics_file, 'w') as f:
                    f.write(f"Backtest für Strategie: {strategy.name}\n")
                    f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Symbol: {args.symbol}, Periode: {args.period}, Intervall: {args.interval}\n\n")
                    f.write("=== Performance-Metriken ===\n")
                    for metric, value in additional_metrics.items():
                        if isinstance(value, float):
                            f.write(f"{metric}: {value:.6f}\n")

                print(f"Detaillierte Metriken gespeichert in {metrics_file}")

            # Ergebnisse plotten und speichern
            fig = backtest_engine.plot_results(
                save_path=os.path.join(args.output_dir, 'charts', f'backtest_results_{args.strategy}.png'))

            # Wenn Visualisierung aktiviert ist, generiere detaillierten Chart mit Signalen
            if args.visualize:
                try:
                    print("\nErstelle Visualisierung mit Handelssignalen...")
                    signals = strategy.generate_signals(data_with_indicators)

                    # Liste von relevanten Indikatoren je nach Strategie
                    if args.strategy == 'mean_reversion':
                        indicators = ['RSI', 'BB_Upper', 'BB_Lower', 'BB_Middle']
                    elif args.strategy == 'volume':
                        indicators = ['SMA_20', 'Volume']
                    elif args.strategy == 'ml':
                        indicators = ['SMA_20', 'EMA_9', 'MACD']
                    elif args.strategy == 'regime':
                        indicators = ['SMA_50', 'ATR', 'RSI']
                    else:
                        indicators = ['SMA_20', 'EMA_9']

                    chart = visualizer.plot_candlestick_with_indicators(
                        data_with_indicators,
                        indicators=indicators,
                        signals=signals
                    )

                    chart_path = os.path.join(args.output_dir, 'charts', f'chart_with_signals_{args.strategy}.html')
                    write_html_with_custom_interaction(chart, chart_path)
                    print(f"Chart mit Signalen und interaktiven Funktionen erstellt und gespeichert unter {chart_path}")
                except Exception as e:
                    print(f"Fehler bei der Visualisierung: {e}")
                    if verbose_level > 0:
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            print(f"Fehler während des Backtests: {e}")
            import traceback
            traceback.print_exc()
# Zusammenfassung und Endergebnis
    end_time = datetime.now()
    execution_time = end_time - datetime.now().replace(microsecond=0)

    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)
    print(f"Symbol: {args.symbol}")
    print(f"Zeitraum: {args.period}")
    print(f"Intervall: {args.interval}")

    # Ausgeführte Aktionen
    actions = []
    if args.repair_data:
        actions.append("Datenreparatur")
    if args.train:
        actions.append("Modelltraining")
    if args.backtest:
        actions.append("Backtest")
    if args.visualize:
        actions.append("Visualisierung")

    if actions:
        print(f"Ausgeführte Aktionen: {', '.join(actions)}")
    else:
        print("Keine Aktionen ausgeführt. Verwende --train, --backtest oder --visualize, um Aktionen auszuführen.")

    # Ausgabeverzeichnisse
    print(f"Ausgabeverzeichnis: {os.path.abspath(args.output_dir)}")

    if args.train:
        model_dir = os.path.join(args.output_dir, 'models')
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        if model_files:
            print(f"Trainierte Modelle: {', '.join(model_files)}")

    if args.backtest:
        metrics_file = os.path.join(args.output_dir, 'reports', f'backtest_metrics_{args.strategy}.txt')
        if os.path.exists(metrics_file):
            print(f"Backtest-Metriken: {metrics_file}")

    print(f"\nAusführungszeit: {execution_time}")
    print("Programm erfolgreich beendet.")

    # Logging abschließen
    with open(os.path.join(args.output_dir, 'execution_log.txt'), 'a') as f:
        f.write(f"Ausführung beendet am: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Ausführungszeit: {execution_time}\n")
        f.write(f"Ausgeführte Aktionen: {', '.join(actions)}\n")
        f.write("-" * 50 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFEHLER: Ein unerwarteter Fehler ist aufgetreten: {e}")
        print("\nStacktrace:")
        import traceback

        traceback.print_exc()

        print("\nBitte überprüfen Sie die Parameter und versuchen Sie es erneut.")
        print("Verwenden Sie --verbose oder -v für mehr Informationen.")
        print("Bei Datenproblemen versuchen Sie --repair-data, um CSV-Dateien zu reparieren.")