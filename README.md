# NQ-Trading-Backtest-Tool

Ein leistungsstarkes Tool zum Backtesting von Handelsstrategien für den Nasdaq-100 Future und andere Märkte.

## Funktionen

- Datenerfassung von Yahoo Finance
- Verschiedene Handelsstrategien (ML, Mean-Reversion, Volume, Combined, Market Regime, Ensemble)
- Machine Learning mit LSTM-Modellen
- Umfassende Backtesting-Engine
- Fortschrittliche Visualisierungen mit interaktiven Charts
- Risikomanagement-Optionen
- Leistungsmetriken und Analyseberichte

## Installation

1. Klone das Repository:
```bash
git clone https://github.com/yourusername/nq-trading-project.git
cd nq-trading-project
```

2. Installiere die erforderlichen Abhängigkeiten:
```bash
pip install -r requirements.txt
```

## Befehlsdokumentation

### Grundlegende Syntax

Alle Befehle werden über die Kommandozeile mit dem `main.py`-Skript ausgeführt:

```bash
python main.py [Optionen]
```

Optionen werden mit `--` gekennzeichnet, gefolgt vom Optionsnamen und ggf. einem Wert.

### Hauptbefehle

Die folgenden Aktionsmodi definieren, was das Tool ausführen soll:

| Befehl | Beschreibung |
|--------|--------------|
| `--train` | Trainiert ein neues ML-Modell |
| `--backtest` | Führt einen Backtest mit der angegebenen Strategie durch |
| `--visualize` | Visualisiert die Daten und Ergebnisse |
| `--repair-data` | Überprüft und repariert Datendateien |

**Hinweis:** Diese Befehle können kombiniert werden, z.B. `--train --backtest --visualize`

### Datenparameter

Diese Parameter steuern, welche Daten verwendet werden:

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--symbol` | Symbol für die zu analysierenden Daten | `NQ=F` (Nasdaq-100 Future) |
| `--period` | Zeitraum für die Daten | `1mo` |
| `--interval` | Datenintervall | `1m` |
| `--custom-file` | Name eines benutzerdefinierten Datensatzes | `None` |
| `--combine-all-years` | Kombiniert alle nq-1m* Dateien | `False` |

#### Verfügbare Perioden (--period)

| Periode | Beschreibung |
|---------|--------------|
| `1d` | 1 Tag |
| `5d` | 5 Tage |
| `1mo` | 1 Monat |
| `3mo` | 3 Monate |
| `6mo` | 6 Monate |
| `1y` | 1 Jahr |
| `2y` | 2 Jahre |
| `5y` | 5 Jahre |
| `10y` | 10 Jahre |
| `max` | Maximaler verfügbarer Zeitraum |

#### Verfügbare Intervalle (--interval)

| Intervall | Beschreibung | Verfügbarkeit |
|-----------|--------------|---------------|
| `1m` | 1 Minute | Zeiträume bis zu 7 Tage |
| `2m` | 2 Minuten | Zeiträume bis zu 60 Tage |
| `5m` | 5 Minuten | Zeiträume bis zu 60 Tage |
| `15m` | 15 Minuten | Zeiträume bis zu 60 Tage |
| `30m` | 30 Minuten | Zeiträume bis zu 60 Tage |
| `60m` | 60 Minuten | Zeiträume bis zu 730 Tage |
| `90m` | 90 Minuten | Zeiträume bis zu 60 Tage |
| `1h` | 1 Stunde (gleich wie 60m) | Zeiträume bis zu 730 Tage |
| `1d` | 1 Tag | Alle Zeiträume |
| `5d` | 5 Tage | Alle Zeiträume |
| `1wk` | 1 Woche | Alle Zeiträume |
| `1mo` | 1 Monat | Alle Zeiträume |
| `3mo` | 3 Monate | Alle Zeiträume |

### Strategieparameter

Diese Parameter steuern die Handelsstrategie:

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--strategy` | Trading-Strategie | `ml` |
| `--risk-management` | Aktiviert Risikomanagement für die Strategie | `False` |

#### Verfügbare Strategien (--strategy)

| Strategie | Beschreibung |
|-----------|--------------|
| `ml` | Machine Learning basierte Strategie |
| `mean_reversion` | Mean-Reversion-Strategie (RSI und Bollinger Bands) |
| `combined` | Kombinierte Strategie (Trend, Momentum, Volatilität) |
| `volume` | Volumen-basierte Strategie |
| `regime` | Market-Regime-Strategie |
| `ensemble` | Ensemble-Strategie (Kombination mehrerer Strategien) |

### Strategiespezifische Parameter

#### Mean-Reversion-Strategie

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--rsi-overbought` | RSI-Grenze für überkauft | `70` |
| `--rsi-oversold` | RSI-Grenze für überverkauft | `30` |

#### Volume-Strategie

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--volume-threshold` | Volumen-Schwellenwert als Vielfaches des Durchschnitts | `1.5` |

#### Kombinierte Strategie

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--trend-weight` | Gewicht für Trend-Signale | `0.4` |
| `--momentum-weight` | Gewicht für Momentum-Signale | `0.3` |
| `--volatility-weight` | Gewicht für Volatilitäts-Signale | `0.3` |

#### Ensemble-Strategie

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--voting-method` | Abstimmungsmethode (majority, unanimous, weighted) | `majority` |

#### Risikomanagement-Parameter

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--risk-per-trade` | Prozentsatz des Kapitals pro Trade | `0.01` (1%) |
| `--position-size-method` | Methode zur Positionsgrößenberechnung (fixed, percent, atr) | `fixed` |
| `--atr-risk-multiplier` | Multiplikator für ATR bei ATR-basierter Positionsgrößenberechnung | `1.5` |
| `--max-drawdown` | Maximaler Drawdown bevor Positionsgröße reduziert wird | `-0.05` (-5%) |

### Backtest-Parameter

Diese Parameter steuern den Backtest:

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--initial-capital` | Anfangskapital für Backtest | `50000` |
| `--commission` | Provisionssatz für Trades | `0.001` (0.1%) |

### ML-Modellparameter

Diese Parameter steuern das Training des ML-Modells:

| Parameter | Beschreibung | Standardwert |
|-----------|--------------|--------------|
| `--window-size` | Fenstergröße für LSTM-Modell | `60` |
| `--epochs` | Anzahl der Trainingsepochen | `50` |
| `--batch-size` | Batch-Größe für Training | `32` |
| `--test-size` | Anteil der Daten für Tests | `0.2` (20%) |

## Beispielbefehle

### Erstmaliger Testlauf

```bash
python main.py --symbol NQ=F --period 1y --interval 1d --visualize
```
Dieser Befehl lädt Nasdaq-100 Future-Daten für ein Jahr im Tagesintervall und visualisiert sie.

### Modell trainieren

```bash
python main.py --symbol NQ=F --period 1mo --interval 1m --train --visualize
```
Dieser Befehl trainiert ein ML-Modell mit Nasdaq-100 Future-Minutendaten über einen Monat und visualisiert die Ergebnisse.

### Backtest mit ML-Strategie

```bash
python main.py --symbol NQ=F --period 1y --interval 1d --backtest --visualize --strategy ml
```
Dieser Befehl führt einen Backtest mit der ML-Strategie auf Nasdaq-100 Future-Tagesdaten über ein Jahr durch und visualisiert die Ergebnisse.

### Mean-Reversion-Strategie mit angepassten RSI-Werten

```bash
python main.py --symbol NQ=F --period 1mo --interval 1m --backtest --visualize --strategy mean_reversion --rsi-overbought 75 --rsi-oversold 25
```
Dieser Befehl führt einen Backtest mit der Mean-Reversion-Strategie durch, wobei die RSI-Grenzen auf 75 (überkauft) und 25 (überverkauft) gesetzt sind.

### Volumen-basierte Strategie

```bash
python main.py --symbol NQ=F --period 1y --interval 1d --backtest --visualize --strategy volume --volume-threshold 1.8
```
Dieser Befehl führt einen Backtest mit der volumenbasierten Strategie durch, wobei der Volumen-Schwellenwert auf das 1,8-fache des Durchschnitts gesetzt ist.

### Strategie mit Risikomanagement

```bash
python main.py --symbol NQ=F --period 1y --interval 1d --backtest --visualize --strategy mean_reversion --risk-management
```
Dieser Befehl führt einen Backtest mit der Mean-Reversion-Strategie und aktiviertem Risikomanagement durch.

### Andere Märkte testen (S&P 500 Futures)

```bash
python main.py --symbol ES=F --period 1y --interval 1d --backtest --visualize --strategy mean_reversion
```
Dieser Befehl führt einen Backtest mit der Mean-Reversion-Strategie auf S&P 500 Future-Daten durch.

### Kurzfristiger Backtest mit Volumen-Strategie

```bash
python main.py --symbol NQ=F --period 3mo --interval 1h --backtest --visualize --strategy volume
```
Dieser Befehl führt einen Backtest mit der volumenbasierten Strategie auf Nasdaq-100 Future-Stundendaten über drei Monate durch.

## Beschränkungen und Inkompatibilitäten

1. **Intervallbeschränkungen für Perioden**:
   - `1m` Intervall ist nur für Zeiträume bis zu 7 Tagen verfügbar
   - `2m`, `5m`, `15m`, `30m`, `90m` Intervalle sind für Zeiträume bis zu 60 Tagen verfügbar
   - `60m`/`1h` Intervall ist für Zeiträume bis zu 730 Tagen verfügbar

2. **Strategieinkompatibilitäten**:
   - Die `ml`-Strategie erfordert ein trainiertes Modell. Führe zuerst `--train` aus, bevor du `--backtest --strategy ml` verwendest.
   - Bei Verwendung von `--strategy ensemble` werden die strategie-spezifischen Parameter (wie `--rsi-overbought`) für die jeweiligen Unterstrategien verwendet.

3. **Datenabhängigkeiten**:
   - Für Minutendaten nutze `--custom-file` oder `--combine-all-years` für umfassende Backtests, da Yahoo Finance nur begrenzte historische Minutendaten bietet.
   - Sehr kurze Zeiträume (weniger als 3x `window-size`) eignen sich nicht für das ML-Modelltraining.

4. **Parameter-Kombinationen**:
   - `--repair-data` kann mit anderen Modi kombiniert werden, wird aber vor allen anderen Operationen ausgeführt.
   - `--train` benötigt genügend Daten, umso mehr bei größerem `--window-size`.
   - `--risk-management` kann mit jeder Strategie kombiniert werden.

## Chart-Interaktionen

Die erzeugten Charts bieten folgende interaktive Funktionen:

- **Mausrad nach unten**: Die X-Achse wird nach rechts gedehnt (quetschen)
- **Mausrad nach oben**: Die X-Achse wird nach links gedehnt (entquetschen)
- **Strg + Mausrad nach oben**: Der Chart wird beidseitig, von der Position des Mauszeigers aus, gedehnt und man bekommt mehr Candlesticks zu sehen
- **Strg + Mausrad nach unten**: Der Chart wird beidseitig in Richtung Mauszeiger gestaucht
- **Klick und Halten auf die Achsen**: Direktes Dehnen der X- oder Y-Achse mit der Maus

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe die [LICENSE](LICENSE) Datei für weitere Details.