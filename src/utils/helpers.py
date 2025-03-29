import pandas as pd


def prepare_dataframe(df):
    """
    Wandelt ein DataFrame mit MultiIndex-Spalten in ein DataFrame mit einfachen Spalten um.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame mit möglichen MultiIndex-Spalten

    Returns:
    --------
    pd.DataFrame
        DataFrame mit einfachen Spalten
    """
    # Kopie erstellen, um das Original nicht zu verändern
    df = df.copy()

    # Prüfen, ob wir MultiIndex-Spalten haben
    if isinstance(df.columns, pd.MultiIndex):
        print("MultiIndex-Spalten erkannt und werden umgewandelt")

        # Extrahiere die ersten Level der Spalten
        ohlc_cols = {'Open': None, 'High': None, 'Low': None, 'Close': None, 'Volume': None}

        # Finde die richtigen Spalten basierend auf dem ersten Level
        for col in df.columns:
            if col[0] in ohlc_cols:
                ohlc_cols[col[0]] = col

        # Erstelle ein neues DataFrame mit den benötigten Spalten
        new_df = pd.DataFrame(index=df.index)
        for simple_name, multi_name in ohlc_cols.items():
            if multi_name is not None:
                new_df[simple_name] = df[multi_name]

        # Füge alle anderen Spalten hinzu (ohne Symbol-Teil)
        for col in df.columns:
            if col[0] not in ohlc_cols and col[0] != '':
                new_df[col[0]] = df[col]

        return new_df
    else:
        # Kein MultiIndex, gebe Original zurück
        return df