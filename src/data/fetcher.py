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

    """
    Kopieren Sie diese Methoden in Ihre DataFetcher-Klasse in src/data/fetcher.py.
    Ersetzen Sie die bestehende load_custom_file-Methode durch diese Version.
    """

    def load_and_standardize_csv(self, file_path):
        """
        Lädt eine CSV-Datei und standardisiert das Format für die weitere Verarbeitung.
        Unterstützt verschiedene Dateiformate:
        1. Yahoo API Format mit einer Datetime-Spalte
        2. Format mit getrennten Datums- und Zeitspalten
        3. Bereits standardisiertes Format mit benannten Spalten

        Args:
            file_path (str): Pfad zur CSV-Datei

        Returns:
            pd.DataFrame: Standardisiertes DataFrame mit 'Date' als Index und OHLCV-Spalten
        """
        import pandas as pd
        import os

        print(f"Lade Datei: {file_path}")

        # Versuche verschiedene Ladestrategien
        try:
            # 1. Versuche zuerst, die Datei zu inspizieren
            with open(file_path, 'r', encoding='utf-8') as f:
                header_lines = min(5, os.path.getsize(file_path) // 100)
                if header_lines == 0:
                    header_lines = 1
                header = [next(f) for _ in range(header_lines)]

            print(f"Datei-Header-Beispiel: {header[0].strip()}")

            # 2. Prüfe auf getrennte Datums- und Zeitspalten (Format 2)
            if ',' in header[0] and len(header[0].split(',')) >= 2:
                first_cols = [h.strip() for h in header[0].split(',')[:2]]

                # Wenn die ersten beiden Spalten wie Datum und Zeit aussehen
                date_like = any(['date' in col.lower() or
                                 '/' in col or
                                 '-' in col or
                                 col.isdigit() for col in first_cols])
                time_like = any(['time' in col.lower() or
                                 ':' in col for col in first_cols])

                if date_like and time_like or (len(first_cols[0]) == 10 and len(first_cols[1]) == 8):
                    print("Format mit getrennten Datums- und Zeitspalten erkannt")
                    df = pd.read_csv(file_path)

                    # Identifiziere Datums- und Zeitspalten
                    date_col = None
                    time_col = None

                    # Suche nach Datumsspalte
                    for col in df.columns:
                        if 'date' in str(col).lower() or df[col].astype(str).str.contains('-').all() or df[col].astype(
                                str).str.contains('/').all():
                            date_col = col
                            break

                    # Wenn keine explizite Datumsspalte gefunden wurde, verwende die erste Spalte
                    if date_col is None and len(df.columns) > 0:
                        date_col = df.columns[0]

                    # Suche nach Zeitspalte
                    for col in df.columns:
                        if 'time' in str(col).lower() or df[col].astype(str).str.contains(':').all():
                            time_col = col
                            break

                    # Wenn keine explizite Zeitspalte gefunden wurde, verwende die zweite Spalte
                    if time_col is None and len(df.columns) > 1:
                        time_col = df.columns[1]

                    print(f"Identifizierte Spalten - Datum: {date_col}, Zeit: {time_col}")

                    if date_col and time_col:
                        # Kombiniere Datums- und Zeitspalten zu Datetime
                        df['Datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str),
                                                        errors='coerce')
                        df.set_index('Datetime', inplace=True)
                        df.index.name = 'Date'

                        # Identifiziere und standardisiere OHLCV-Spalten
                        remaining_cols = [c for c in df.columns if c != date_col and c != time_col]

                        # Wenn genau 5 verbleibende Spalten vorhanden sind, nehme an, dass es OHLCV ist
                        if len(remaining_cols) >= 5:
                            # Prüfe, ob die Spalten bereits benannt sind
                            ohlcv_names = {'open', 'high', 'low', 'close', 'volume'}
                            existing_ohlcv = [c for c in df.columns if c.lower() in ohlcv_names]

                            if len(existing_ohlcv) >= 4:  # Wenn die meisten OHLCV-Spalten bereits benannt sind
                                print("OHLCV-Spaltennamen bereits vorhanden")
                            else:
                                # Benenne die verbleibenden numerischen Spalten als OHLCV
                                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                                if len(numeric_cols) >= 5:
                                    rename_dict = {
                                        numeric_cols[0]: 'Open',
                                        numeric_cols[1]: 'High',
                                        numeric_cols[2]: 'Low',
                                        numeric_cols[3]: 'Close',
                                        numeric_cols[4]: 'Volume'
                                    }
                                    df.rename(columns=rename_dict, inplace=True)
                                    print(f"Spalten umbenannt: {rename_dict}")

                    return df

            # 3. Versuche mit Standardoptionen (Format 1 oder 3)
            try:
                # Versuche mit 'Date' als Index
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                print("Erfolgreich mit 'Date' als Index geladen")

                # Prüfe, ob OHLCV-Spalten fehlen
                if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    # Versuche, Spalten zu identifizieren und umzubenennen
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if len(numeric_cols) >= 4:  # Mindestens OHLC
                        rename_dict = {}
                        ohlcv = ['Open', 'High', 'Low', 'Close']
                        for i, col in enumerate(numeric_cols[:4]):
                            rename_dict[col] = ohlcv[i]

                        if len(numeric_cols) >= 5:
                            rename_dict[numeric_cols[4]] = 'Volume'

                        df.rename(columns=rename_dict, inplace=True)
                        print(f"Spalten umbenannt: {rename_dict}")

                return df
            except:
                # Wenn kein 'Date' als Index, versuche mit Standardoptionen
                df = pd.read_csv(file_path)

                # Prüfe auf Datetime-Spalte
                for col in df.columns:
                    if 'date' in str(col).lower() or 'time' in str(col).lower() or 'datetime' in str(col).lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                            df.set_index(col, inplace=True)
                            df.index.name = 'Date'
                            print(f"Erfolgreich mit '{col}' als Datetime-Index geladen")
                            break
                        except:
                            continue

                # Wenn kein erfolgreicher Index gesetzt wurde, versuche die erste Spalte
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.set_index(df.columns[0], inplace=True)
                        df.index = pd.to_datetime(df.index)
                        df.index.name = 'Date'
                        print("Erste Spalte als Datetime-Index verwendet")
                    except:
                        print("Konnte keinen Datetime-Index setzen")

                # Prüfe, ob OHLCV-Spalten fehlen
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in expected_cols):
                    # Prüfe auf bekannte Spaltennamenvarianten
                    col_variants = {
                        'Open': ['open', 'o', 'opening', 'first'],
                        'High': ['high', 'h', 'max', 'highest'],
                        'Low': ['low', 'l', 'min', 'lowest'],
                        'Close': ['close', 'c', 'last', 'closing'],
                        'Volume': ['volume', 'vol', 'v']
                    }

                    rename_dict = {}
                    for std_col, variants in col_variants.items():
                        if std_col not in df.columns:
                            for var in variants:
                                matching_cols = [c for c in df.columns if var == c.lower()]
                                if matching_cols:
                                    rename_dict[matching_cols[0]] = std_col
                                    break

                    if rename_dict:
                        df.rename(columns=rename_dict, inplace=True)
                        print(f"Spalten umbenannt: {rename_dict}")

                return df

        except Exception as e:
            print(f"Fehler beim Laden der Datei: {e}")
            # Fallback mit verschiedenen skiprows-Werten probieren
            for skiprows in range(0, 5):
                try:
                    df = pd.read_csv(file_path, skiprows=skiprows)
                    print(f"Erfolgreich mit skiprows={skiprows} geladen")

                    # Wenn erfolgreich, versuche die Datei zu standardisieren
                    return self.standardize_dataframe(df)
                except:
                    continue

            # Wenn alles fehlschlägt, leere DataFrame zurückgeben
            print("Konnte Datei nicht laden, gebe leeres DataFrame zurück")
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    def standardize_dataframe(self, df):
        """
        Standardisiert ein DataFrame für die weitere Verarbeitung.

        Args:
            df (pd.DataFrame): Original DataFrame

        Returns:
            pd.DataFrame: Standardisiertes DataFrame mit 'Date' als Index und OHLCV-Spalten
        """
        import pandas as pd

        # Kopie erstellen
        df = df.copy()

        # 1. Stelle sicher, dass wir einen Datetime-Index haben
        if not isinstance(df.index, pd.DatetimeIndex):
            # Suche nach Datetime-Spalten
            datetime_cols = []
            for col in df.columns:
                if 'date' in str(col).lower() or 'time' in str(col).lower() or 'datetime' in str(col).lower():
                    datetime_cols.append(col)

            # Wenn Datetime-Spalten gefunden wurden, verwende die erste
            if datetime_cols:
                try:
                    df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]])
                    df.set_index(datetime_cols[0], inplace=True)
                    df.index.name = 'Date'
                except:
                    print(f"Konnte {datetime_cols[0]} nicht als Datetime-Index verwenden")

        # 2. Stelle sicher, dass OHLCV-Spalten vorhanden sind
        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if missing_cols:
            print(f"Fehlende Spalten: {missing_cols}")

            # Wenn es genau die richtige Anzahl numerischer Spalten gibt, benenne sie um
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) >= len(missing_cols):
                for i, col in enumerate(missing_cols):
                    if i < len(numeric_cols):
                        df[col] = df[numeric_cols[i]]
                        print(f"Spalte {numeric_cols[i]} als {col} verwendet")

        # 3. Füge fehlende Spalten mit NaN-Werten hinzu
        for col in expected_cols:
            if col not in df.columns:
                df[col] = float('nan')
                print(f"Spalte {col} mit NaN-Werten hinzugefügt")

        return df

    def load_custom_file(self, file_pattern):
        """
        Lädt Daten aus einer benutzerdefinierten Datei oder einem Muster.
        Unterstützt verschiedene Dateiformate, einschließlich:
        - Yahoo API Format mit einer Datetime-Spalte
        - Format mit getrennten Datums- und Zeitspalten
        - Bereits standardisiertes Format mit benannten Spalten

        Args:
            file_pattern (str): Dateiname oder Glob-Muster für die zu ladenden Dateien

        Returns:
            pandas.DataFrame: DataFrame mit den geladenen Daten
        """
        import os
        import pandas as pd
        import glob

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
                    df = self.load_and_standardize_csv(file)

                    if not df.empty:
                        all_dfs.append(df)
                        print(f"Datei {file} erfolgreich geladen, Shape: {df.shape}")
                    else:
                        print(f"Datei {file} konnte nicht geladen werden oder ist leer")
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
            return self.load_and_standardize_csv(matching_files[0])
    # Diese Funktion in DataFetcher integrieren

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