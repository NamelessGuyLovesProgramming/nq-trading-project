import pandas as pd
from src.utils.helpers import prepare_dataframe

class BaseStrategy:
    def __init__(self, name="BaseStrategy"):
        """
        Basis-Klasse für alle Trading-Strategien.

        Parameters:
        -----------
        name : str
            Name der Strategie
        """
        self.name = name
        self.positions = []
        self.current_position = 0  # 0: kein, 1: long, -1: short

    def generate_signals(self, data):
        """
        Generiert Trading-Signale basierend auf den Daten.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame mit OHLCV- und Indikatordaten

        Returns:
        --------
        pd.DataFrame
            DataFrame mit zusätzlicher Signalspalte
        """
        raise NotImplementedError("Diese Methode muss in der Unterklasse implementiert werden")

    def calculate_returns(self, data, signals, initial_capital=10000.0, commission=0.001):
        """
        Berechnet die Renditen der Strategie.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame mit OHLCV-Daten
        signals : pd.DataFrame
            DataFrame mit Signalen
        initial_capital : float
            Anfangskapital
        commission : float
            Kommissionssatz

        Returns:
        --------
        dict
            Dictionary mit Backtest-Ergebnissen
        """
        # Daten für die Verarbeitung vorbereiten
        backtest = prepare_dataframe(data.copy())
        signals = prepare_dataframe(signals)

        # Sicherstellen, dass 'Signal' korrekt ist (ein Series, keine DataFrame)
        if isinstance(signals['Signal'], pd.DataFrame):
            backtest['Signal'] = signals['Signal'].iloc[:, 0]
        else:
            backtest['Signal'] = signals['Signal']

        # Berechne tägliche Preisänderungen
        backtest['Returns'] = backtest['Close'].pct_change()

        # Strategie-Renditen (verzögert um einen Tag)
        backtest['Strategy'] = backtest['Signal'].shift(1) * backtest['Returns']

        # Berücksichtige Kommissionen
        # Sicherstellen, dass wir mit Series arbeiten, keine DataFrames
        trades = backtest['Signal'].diff().abs()
        if isinstance(trades, pd.DataFrame):
            trades = trades.iloc[:, 0]

        close_series = backtest['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]

        backtest['Trades'] = trades
        backtest['Commission'] = trades * commission * close_series

        # Gesamtrendite mit Kommissionen
        strategy_series = backtest['Strategy']
        if isinstance(strategy_series, pd.DataFrame):
            strategy_series = strategy_series.iloc[:, 0]

        commission_series = backtest['Commission']
        if isinstance(commission_series, pd.DataFrame):
            commission_series = commission_series.iloc[:, 0]

        backtest['Strategy_Net'] = strategy_series * close_series - commission_series

        # Kumuliere Renditen
        backtest['Cumulative_Returns'] = (1 + backtest['Returns']).cumprod()
        backtest['Cumulative_Strategy'] = (1 + backtest['Strategy']).cumprod()

        # Portfolio-Werte
        backtest['Portfolio_Value'] = initial_capital * backtest['Cumulative_Strategy']

        # Drawdown-Berechnung
        backtest['High_Value'] = backtest['Portfolio_Value'].cummax()
        backtest['Drawdown'] = (backtest['Portfolio_Value'] - backtest['High_Value']) / backtest['High_Value'] * 100

        # Ergebnis-Dictionary
        results = {
            'portfolio_value': backtest['Portfolio_Value'],
            'returns': backtest['Strategy'],
            'cumulative_returns': backtest['Cumulative_Strategy'],
            'drawdown': backtest['Drawdown'],
            'trades': backtest['Trades'].sum() / 2,  # Teile durch 2, da jeder Trade zwei Einträge hat (ein/aus)
            'win_rate': None,  # Wird später berechnet
            'sharpe_ratio': None,  # Wird später berechnet
            'max_drawdown': backtest['Drawdown'].min()
        }

        # Berechne zusätzliche Metriken
        annualized_return = (backtest['Cumulative_Strategy'].iloc[-1] ** (252 / len(backtest))) - 1
        annualized_volatility = backtest['Strategy'].std() * (252 ** 0.5)

        # Sharpe Ratio (annualisiert, risikofrei = 0 für Einfachheit)
        if annualized_volatility != 0:
            results['sharpe_ratio'] = annualized_return / annualized_volatility

        # Gewinnrate
        trades = []
        current_position = 0
        entry_price = 0

        for i, row in backtest.iterrows():
            # Extrahiere Signal-Wert als skalaren Wert (falls es eine Series ist)
            signal_value = row['Signal']
            if isinstance(signal_value, pd.Series):
                signal_value = signal_value.iloc[0]

            # Extrahiere Close-Wert als skalaren Wert (falls es eine Series ist)
            close_value = row['Close']
            if isinstance(close_value, pd.Series):
                close_value = close_value.iloc[0]

            if signal_value != current_position and signal_value != 0:
                if current_position != 0:  # Schließe Position
                    pnl = (close_value - entry_price) * current_position
                    trades.append(pnl)

                current_position = signal_value
                entry_price = close_value

        if len(trades) > 0:
            results['win_rate'] = sum(1 for trade in trades if trade > 0) / len(trades)

        return results