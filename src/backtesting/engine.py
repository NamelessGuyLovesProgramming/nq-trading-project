import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os


class BacktestEngine:
    def __init__(self, data, strategy, initial_capital=10000.0, commission=0.001):
        """
        Backtest-Engine für die Strategieevaluierung.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame mit OHLCV-Daten
        strategy : Strategy
            Trading-Strategie-Objekt
        initial_capital : float
            Anfangskapital
        commission : float
            Kommissionssatz
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = None

    def run(self):
        """
        Führt den Backtest durch.

        Returns:
        --------
        dict
            Dictionary mit Backtest-Ergebnissen
        """
        # Generiere Signale
        signals = self.strategy.generate_signals(self.data)

        # Berechne Renditen
        self.results = self.strategy.calculate_returns(
            self.data, signals, self.initial_capital, self.commission
        )

        # Ausgabe der Ergebnisse
        self._print_results()

        return self.results

    def _print_results(self):
        """
        Gibt die Backtest-Ergebnisse aus.
        """
        if self.results is None:
            print("Es wurden noch keine Backtest-Ergebnisse generiert.")
            return

        print(f"\n=== Backtest-Ergebnisse für {self.strategy.name} ===")
        print(f"Anfangskapital: ${self.initial_capital:.2f}")
        print(f"Endkapital: ${self.results['portfolio_value'].iloc[-1]:.2f}")
        print(f"Gesamtrendite: {(self.results['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100:.2f}%")
        print(f"Maximaler Drawdown: {self.results['max_drawdown']:.2f}%")
        print(f"Anzahl der Trades: {self.results['trades']}")

        if self.results['win_rate'] is not None:
            print(f"Gewinnrate: {self.results['win_rate'] * 100:.2f}%")

        if self.results['sharpe_ratio'] is not None:
            print(f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")

    def plot_results(self, benchmark=None, save_path=None):
        """
        Plottet die Backtest-Ergebnisse.

        Parameters:
        -----------
        benchmark : pd.Series
            Benchmark-Performance zum Vergleich
        save_path : str
            Pfad zum Speichern des Plots

        Returns:
        --------
        matplotlib.figure.Figure
            Matplotlib-Figure-Objekt
        """
        if self.results is None:
            print("Es wurden noch keine Backtest-Ergebnisse generiert.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

        # Portfolio-Wert
        self.results['portfolio_value'].plot(ax=axes[0], label=self.strategy.name)

        # Benchmark, falls vorhanden
        if benchmark is not None:
            benchmark = benchmark / benchmark.iloc[0] * self.initial_capital
            benchmark.plot(ax=axes[0], label='Benchmark', alpha=0.7)

        axes[0].set_title('Backtest-Ergebnisse')
        axes[0].set_ylabel('Portfolio-Wert ($)')
        axes[0].legend()
        axes[0].grid(True)

        # Drawdown
        self.results['drawdown'].plot(ax=axes[1], color='red', alpha=0.7)
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].set_xlabel('Datum')
        axes[1].grid(True)

        # Füge wichtige Metriken als Text hinzu
        metrics_text = (
            f"Gesamtrendite: {(self.results['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100:.2f}%\n"
            f"Max. Drawdown: {self.results['max_drawdown']:.2f}%\n"
            f"Trades: {self.results['trades']}\n"
        )

        if self.results['win_rate'] is not None:
            metrics_text += f"Gewinnrate: {self.results['win_rate'] * 100:.2f}%\n"

        if self.results['sharpe_ratio'] is not None:
            metrics_text += f"Sharpe Ratio: {self.results['sharpe_ratio']:.2f}"

        axes[0].annotate(metrics_text, xy=(0.02, 0.05), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Plot gespeichert unter: {save_path}")

        return fig