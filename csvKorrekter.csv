import pandas as pd

# Lade die Datei
df = pd.read_csv('nq-1m2023.csv')

# Kombiniere Datum und Zeit in eine Datetime-Spalte
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Zeit'])

# Setze Datetime als Index und entferne die alten Spalten
df = df.set_index('Datetime')
df = df.drop(columns=['Date', 'Zeit'])

# Speichere die neue Datei
df.to_csv('nq-1m2023-fixed.csv')