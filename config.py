# Konfigurationsparameter für NQ Trading Backtest Tool

# Grundlegende Datenparameter
DEFAULT_SYMBOL = "NQ=F"  # Nasdaq-100 Future
DEFAULT_PERIOD = "1mo"    # 1 Monat
DEFAULT_INTERVAL = "1m"  # Minuten Daten

# Parameter für Mean-Reversion-Strategie
RSI_OVERBOUGHT = 70      # RSI-Grenze für überkauft
RSI_OVERSOLD = 30        # RSI-Grenze für überverkauft

# Parameter für Volume-Strategie
VOLUME_THRESHOLD = 1.5   # Volumen-Schwellenwert als Vielfaches des Durchschnitts

# Parameter für Backtest
INITIAL_CAPITAL = 50000  # Startkapital
COMMISSION = 0.001        # Provision als Prozentsatz (0.1%)

# Parameter für ML-Modell
WINDOW_SIZE = 60         # Fenstergröße für Sequenzdaten
BATCH_SIZE = 32          # Batch-Größe für Training
EPOCHS = 50              # Anzahl der Trainingsepochen

# Pfadkonfigurationen
OUTPUT_DIR = "output"    # Ausgabeverzeichnis
DATA_DIR = "data/raw"    # Verzeichnis für Rohdaten