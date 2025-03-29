# src/strategies/ensemble.py
import pandas as pd
import numpy as np
from src.strategies.base import BaseStrategy
from src.utils.helpers import prepare_dataframe


class EnsembleStrategy(BaseStrategy):
    def __init__(self, strategies, voting_method='majority'):
        """
        Ensemble von Strategien mit verschiedenen Voting-Methoden.

        Parameters:
        -----------
        strategies : list
            Liste von Strategie-Instanzen
        voting_method : str
            Wie die Signale kombiniert werden ('majority', 'weighted', 'unanimous')
        """
        super().__init__(name="Ensemble_Strategy")
        self.strategies = strategies
        self.voting_method = voting_method

    def generate_signals(self, data):
        # Daten für die Verarbeitung vorbereiten
        df = prepare_dataframe(data.copy())

        # Sammle Signale von allen Strategien
        all_signals = pd.DataFrame(index=df.index)
        for i, strategy in enumerate(self.strategies):
            strategy_signals = strategy.generate_signals(df)
            all_signals[f'Strategy_{i}'] = strategy_signals['Signal']

        # Kombiniere Signale basierend auf der Voting-Methode
        if self.voting_method == 'majority':
            # Mehrheitsentscheidung
            all_signals['Sum'] = all_signals.sum(axis=1)
            df['Signal'] = 0
            df.loc[all_signals['Sum'] > 0, 'Signal'] = 1
            df.loc[all_signals['Sum'] < 0, 'Signal'] = -1

        elif self.voting_method == 'unanimous':
            # Einstimmige Entscheidung
            df['Signal'] = 0
            all_positive = (all_signals > 0).all(axis=1)
            all_negative = (all_signals < 0).all(axis=1)
            df.loc[all_positive, 'Signal'] = 1
            df.loc[all_negative, 'Signal'] = -1

        elif self.voting_method == 'weighted':
            # Gewichtete Entscheidung (mit gleichen Gewichten)
            weights = [1 / len(self.strategies)] * len(self.strategies)
            df['Signal'] = 0
            weighted_sum = (all_signals * weights).sum(axis=1)
            df.loc[weighted_sum > 0.3, 'Signal'] = 1
            df.loc[weighted_sum < -0.3, 'Signal'] = -1

        return df