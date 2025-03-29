def repair_csv_files(data_dir='data/raw', force_download=False):
    """
    Überprüft und repariert CSV-Dateien im angegebenen Verzeichnis,
    die ein spezielles Format mit mehreren Header-Zeilen haben.

    Args:
        data_dir (str): Verzeichnis mit den CSV-Dateien
        force_download (bool): Wenn True, werden alle Dateien neu formatiert
    """
    import os
    import pandas as pd
    import glob
    import shutil

    print(f"Überprüfe Dateien im Verzeichnis {data_dir}")

    # Stelle sicher, dass das Verzeichnis existiert
    os.makedirs(data_dir, exist_ok=True)

    # Finde alle CSV-Dateien im Verzeichnis
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Gefundene CSV-Dateien: {len(csv_files)}")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"Verarbeite: {file_name}")

        try:
            # Lese die ersten Zeilen der Datei, um das Format zu erkennen
            with open(file_path, 'r') as f:
                header_lines = [next(f) for _ in range(5)]

            print("Erste Zeilen:")
            for i, line in enumerate(header_lines):
                print(f"  Zeile {i}: {line.strip()}")

            # Erstelle Backup, falls noch nicht vorhanden
            backup_path = file_path + ".bak"
            if not os.path.exists(backup_path) or force_download:
                shutil.copy2(file_path, backup_path)
                print(f"Original gesichert unter {backup_path}")

            # Versuche, die Datei mit verschiedenen Datumsformaten zu lesen
            success = False

            # Versuch 1: Europäisches Format (DD/MM/YYYY)
            try:
                df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
                print(f"Datei erfolgreich mit europäischem Datumsformat gelesen. Shape: {df.shape}")
                success = True
            except Exception as e:
                print(f"Fehler beim Lesen mit europäischem Datumsformat: {e}")

                # Versuch 2: Probiere mit skiprows
                for i in range(1, 5):
                    try:
                        df = pd.read_csv(file_path, skiprows=i, parse_dates=['Date'], dayfirst=True)
                        if 'Date' in df.columns:
                            print(
                                f"Erfolgreich mit skiprows={i} und europäischem Datumsformat gelesen. Shape: {df.shape}")
                            success = True
                            break
                    except Exception as skip_e:
                        continue

            # Wenn immer noch kein Erfolg, versuche andere Datumsformate
            if not success:
                try:
                    df = pd.read_csv(file_path, parse_dates=['Date'], format='mixed')
                    print(f"Datei erfolgreich mit gemischtem Datumsformat gelesen. Shape: {df.shape}")
                    success = True
                except Exception as e:
                    print(f"Fehler beim Lesen mit gemischtem Datumsformat: {e}")

            # Wenn erfolgreich, speichere die Datei neu
            if success:
                # Stelle sicher, dass das Datum richtig formatiert ist
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])

                # Speichere die Datei neu
                df.to_csv(file_path, index=False)
                print("Datei erfolgreich neu formatiert und gespeichert.")
            else:
                print("Konnte keine geeignete Formatierung finden. Datei bleibt unverändert.")

        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei: {e}")