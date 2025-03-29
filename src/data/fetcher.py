import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
import glob


class DataFetcher:
    """
    Klasse zum Herunterladen und Laden von Finanzdaten
    """

    def __init__(self, symbol, data_dir='data/raw', cache_days=7):
        """
        Initialisieren des DataFetcher mit Symbol und Datenverzeichnis

        Args:
            symbol (str): Das zu betrachtende Symbol, z.B. 'NQ=F' für Nasdaq Future
            data_dir (str): Verzeichnis zum Speichern der heruntergeladenen Daten
            cache_days (int): Anzahl der Tage, bevor gecachte Daten als veraltet betrachtet werden
        """
        self.symbol = symbol
        self.data_dir = data_dir
        self.cache_days = cache_days  # Neue Eigenschaft für Cache-Dauer

        # Erstelle Verzeichnis, falls es nicht existiert
        os.makedirs(data_dir, exist_ok=True)

        # Logging einrichten
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, period='1y', interval='1d', force_download=False, custom_file=None):
        """
        Lade Daten für das angegebene Symbol. Falls die Daten lokal vorhanden sind,
        werden sie von dort geladen, es sei denn, force_download ist True.

        Args:
            period (str): Zeitraum für die Daten (z.B. '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
            interval (str): Intervall der Daten (z.B. '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            force_download (bool): Wenn True, werden die Daten neu heruntergeladen, auch wenn sie lokal existieren
            custom_file (str): Wenn angegeben, wird versucht, die Daten aus dieser Datei zu laden

        Returns:
            pandas.DataFrame: DataFrame mit den geladenen Daten oder leerer DataFrame bei Fehler
        """
        try:
            # Neue Funktion: Wenn ein benutzerdefinierter Dateiname angegeben wird
            if custom_file:
                return self.load_custom_file(custom_file)

            # Für 1-Minuten-Daten verwende spezielle Methode, da Yahoo nur die letzten 30 Tage unterstützt
            if interval == '1m' and period.startswith('nq-1m'):
                # Neue Funktion für die nq-1m* Dateien
                return self.load_nq_minute_data(period)
            elif interval == '1m':
                return self.fetch_minute_data(period, force_download)

            # Normaler Ablauf für alle anderen Intervalle
            # Dateipfad für lokale Datenspeicherung
            file_name = f"{self.symbol}_{interval}_{period}.csv"
            file_path = os.path.join(self.data_dir, file_name)

            # Prüfe, ob Datei existiert und ob sie aktuell ist
            if os.path.exists(file_path) and not force_download:
                try:
                    # Überprüfe Dateiattribute
                    file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))).days

                    # Wenn Datei älter als cache_days und period != max, dann neu herunterladen
                    # Verwende die neue cache_days Eigenschaft statt fest codierter 1 Tag
                    if file_age_days > self.cache_days and period != 'max':
                        print(
                            f"Datei ist {file_age_days} Tage alt (Cache-Dauer: {self.cache_days} Tage). Lade neu herunter...")
                        force_download = True
                    else:
                        print(
                            f"Lade Daten aus lokaler Datei: {file_path} (Alter: {file_age_days} Tage, noch gültig für {max(0, self.cache_days - file_age_days)} Tage)")
                        try:
                            # Versuche zuerst mit Standard-Parametern
                            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                            if df.empty:
                                raise ValueError("Datei ist leer")
                            return df
                        except Exception as e:
                            print(f"Fehler beim Laden der Datei: {e}")
                            print("Versuche Datei mit speziellen Parametern zu laden...")
                            # Versuche mit speziellen Parametern für das besondere Format
                            df = pd.read_csv(file_path, skiprows=3)
                            if df.empty:
                                raise ValueError("Datei ist leer")
                            df.set_index("Date", inplace=True)
                            return df
                except Exception as e:
                    print(f"Fehler beim Überprüfen/Laden der Datei: {e}")
                    force_download = True

            # Wenn force_download True ist oder Datei nicht existiert, lade von yfinance
            if force_download or not os.path.exists(file_path):
                print(f"Lade Daten von Yahoo Finance: {self.symbol} ({period}, {interval})")
                try:
                    # Lade Daten mit yfinance
                    data = yf.download(self.symbol, period=period, interval=interval)

                    # Überprüfe, ob Daten erfolgreich geladen wurden
                    if data.empty:
                        print(f"Keine Daten für {self.symbol} gefunden. Überprüfe Symbol und Parameter.")
                        # Erstelle leeren DataFrame mit den erwarteten Spalten
                        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

                    # Speichere Daten lokal
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    data.to_csv(file_path)
                    print(f"Daten gespeichert nach: {file_path}")
                    return data
                except Exception as e:
                    print(f"Fehler beim Herunterladen der Daten: {e}")
                    # Versuche lokale Daten zu laden, falls vorhanden
                    if os.path.exists(file_path):
                        print("Versuche stattdessen lokale Datei zu laden...")
                        try:
                            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                            if not df.empty:
                                return df
                        except:
                            pass

                    # Wenn alles fehlschlägt, erstelle leeren DataFrame
                    print("Konnte keine Daten laden. Erstelle leeren DataFrame.")
                    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        except Exception as e:
            print(f"Unerwarteter Fehler beim Laden der Daten: {e}")
            import traceback
            traceback.print_exc()
            # Erstelle leeren DataFrame als Fallback
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    def load_custom_file(self, file_pattern):
        """
        Lädt Daten aus einer benutzerdefinierten Datei oder einem Muster.

        Args:
            file_pattern (str): Dateiname oder Glob-Muster für die zu ladenden Dateien

        Returns:
            pandas.DataFrame: DataFrame mit den geladenen Daten
        """
        file_path = os.path.join(self.data_dir, file_pattern)

        # Prüfe, ob die Datei existiert oder ob es ein Glob-Muster ist
        matching_files = glob.glob(file_path)

        if not matching_files:
            print(f"Keine Dateien gefunden, die dem Muster '{file_path}' entsprechen.")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        print(f"Gefundene Dateien: {matching_files}")

        # Wenn mehrere Dateien gefunden wurden, kombiniere sie
        if len(matching_files) > 1:
            print(f"Lade und kombiniere {len(matching_files)} Dateien...")
            all_dfs = []

            for file in sorted(matching_files):
                try:
                    print(f"Lade Datei: {file}")
                    df = pd.read_csv(file)

                    # Überprüfe auf notwendige Spalten
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
                    missing_cols = [col for col in required_cols if col not in df.columns]

                    if missing_cols:
                        print(f"Warnung: Fehlende Spalten in {file}: {missing_cols}")
                        # Versuche alternative Spaltennamen zu finden
                        if 'Date' in missing_cols and 'Datetime' in df.columns:
                            df.rename(columns={'Datetime': 'Date'}, inplace=True)
                            missing_cols.remove('Date')

                        # Falls immer noch Spalten fehlen, überspringe die Datei
                        if missing_cols:
                            print(f"Überspringe Datei {file} wegen fehlender Spalten")
                            continue

                    # Setze Datum als Index, falls noch nicht geschehen
                    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df.index):
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)

                    all_dfs.append(df)
                    print(f"Datei {file} erfolgreich geladen, Shape: {df.shape}")
                except Exception as e:
                    print(f"Fehler beim Laden der Datei {file}: {e}")

            if not all_dfs:
                print("Keine Daten konnten geladen werden.")
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

            # Kombiniere alle DataFrames
            combined_df = pd.concat(all_dfs)

            # Entferne Duplikate
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

            # Sortiere nach Datum
            combined_df.sort_index(inplace=True)

            print(f"Kombinierte Daten: {len(combined_df)} Einträge")
            return combined_df
        else:
            # Nur eine Datei gefunden
            file_path = matching_files[0]
            print(f"Lade Datei: {file_path}")

            try:
                # Versuche verschiedene Methoden zum Laden der Datei
                try:
                    # Standardmethode
                    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                except:
                    try:
                        # Alternative Methode ohne Index
                        df = pd.read_csv(file_path)
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)
                        elif 'Datetime' in df.columns:
                            df['Datetime'] = pd.to_datetime(df['Datetime'])
                            df.set_index('Datetime', inplace=True)
                            df.index.name = 'Date'
                    except:
                        # Methode mit skiprows für Yahoo-Format
                        df = pd.read_csv(file_path, skiprows=3)
                        if 'Date' in df.columns:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df.set_index('Date', inplace=True)

                print(f"Datei erfolgreich geladen, Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"Fehler beim Laden der Datei {file_path}: {e}")
                import traceback
                traceback.print_exc()
                return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    def load_nq_minute_data(self, period):
        """
        Lädt die NQ 1-Minuten-Daten aus den nq-1m* Dateien.

        Args:
            period (str): Der spezifische Zeitraum, z.B. 'nq-1m2020' bis 'nq-1m2025'

        Returns:
            pandas.DataFrame: DataFrame mit den geladenen Daten
        """
        # Prüfe, ob es eine exakte Übereinstimmung gibt
        file_pattern = f"{period}.csv"
        file_path = os.path.join(self.data_dir, file_pattern)

        if os.path.exists(file_path):
            print(f"Lade spezifische NQ 1-Minuten-Datei: {file_path}")
            try:
                df = pd.read_csv(file_path)

                # Stelle sicher, dass eine Datumsspalte existiert und als Index gesetzt ist
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                elif 'Datetime' in df.columns:
                    df['Datetime'] = pd.to_datetime(df['Datetime'])
                    df.set_index('Datetime', inplace=True)
                    df.index.name = 'Date'

                print(f"Datei erfolgreich geladen, Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"Fehler beim Laden der Datei {file_path}: {e}")

        # Wenn keine exakte Übereinstimmung, versuche ein Muster
        file_pattern = "nq-1m*.csv"
        return self.load_custom_file(file_pattern)

    def fetch_minute_data(self, period, force_download=False):
        """
        Lädt 1-Minuten-Daten mit Berücksichtigung der Yahoo Finance Einschränkung
        (nur die letzten 30 Tage verfügbar).

        Args:
            period (str): Zeitraum für die Daten (z.B. '1mo', '3mo', '6mo', '1y')
            force_download (bool): Wenn True, werden die Daten neu heruntergeladen

        Returns:
            pandas.DataFrame: DataFrame mit 1-Minuten-Daten
        """
        # Prüfe, ob bereits eine kombinierte Datei existiert
        file_name = f"{self.symbol}_1m_{period}_combined.csv"
        file_path = os.path.join(self.data_dir, file_name)

        if os.path.exists(file_path) and not force_download:
            file_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))).days

            # Nur wiederverwenden, wenn die Datei innerhalb der Cache-Dauer ist
            # Verwende die neue cache_days Eigenschaft statt fest codierter 1 Tag
            if file_age_days <= self.cache_days:
                self.logger.info(
                    f"Lade 1-Minuten-Daten aus lokaler Datei: {file_path} (Alter: {file_age_days} Tage, noch gültig für {max(0, self.cache_days - file_age_days)} Tage)")
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    if not df.empty:
                        return df
                except Exception as e:
                    self.logger.warning(f"Fehler beim Laden der Datei: {e}")
                    # Weiter zum Neuladen
            else:
                self.logger.info(f"Datei ist {file_age_days} Tage alt (Cache-Dauer: {self.cache_days} Tage). Lade neu.")

        # Berechnen des maximalen Zeitraums (max. 30 Tage für 1m-Daten)
        end_date = datetime.now()

        # Für 1-Minuten-Daten beschränken wir auf maximal 30 Tage
        # unabhängig vom angeforderten Zeitraum
        start_date = end_date - timedelta(days=30)

        self.logger.info(f"Lade 1-Minuten-Daten für die letzten 30 Tage: {start_date.date()} bis {end_date.date()}")
        self.logger.info("Hinweis: Yahoo Finance begrenzt 1-Minuten-Daten auf die letzten 30 Tage")

        # Optimierte Chunk-Größe für 1-Minuten-Daten
        # Erhöhe die Chunk-Größe von 5 auf 7 Tage, um Downloads zu reduzieren
        chunk_size = 7  # 7 Tage pro Chunk statt 5

        # Teile den Zeitraum in Chunks auf
        chunk_data = []
        current_date = start_date

        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=chunk_size), end_date)

            # Setze das Startdatum und Enddatum für yfinance
            start_str = current_date.strftime('%Y-%m-%d')
            end_str = chunk_end.strftime('%Y-%m-%d')

            self.logger.info(f"Lade Chunk: {start_str} bis {end_str}")

            try:
                # Lade die Daten für diesen Chunk
                chunk_df = yf.download(self.symbol, start=start_str, end=end_str, interval='1m')

                if not chunk_df.empty:
                    self.logger.info(f"Chunk geladen: {len(chunk_df)} Einträge")
                    chunk_data.append(chunk_df)
                else:
                    self.logger.warning(f"Keine Daten für Chunk {start_str} bis {end_str}")

                # Warte kurz zwischen den Anfragen, um API-Limits zu respektieren
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Fehler beim Laden des Chunks {start_str} bis {end_str}: {e}")
                # Bei bestimmten Fehlern abbrechen (z.B. Daten nicht verfügbar)
                if "data not available" in str(e):
                    self.logger.warning("Daten nicht verfügbar für diesen Zeitraum, breche weitere Chunk-Anfragen ab")
                    break

            # Gehe zum nächsten Chunk
            current_date = chunk_end + timedelta(days=0.1)  # Kleiner Offset um Überlappungen zu vermeiden

        # Kombiniere alle Chunks
        if not chunk_data:
            self.logger.error("Keine Daten konnten geladen werden")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        combined_df = pd.concat(chunk_data)

        # Entferne Duplikate (können durch Überlappungen entstehen)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Sortiere nach Datum
        combined_df.sort_index(inplace=True)

        # Speichere die kombinierten Daten
        combined_df.to_csv(file_path)
        self.logger.info(f"Kombinierte Daten gespeichert nach: {file_path}")

        return combined_df