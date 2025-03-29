# src/backtesting/metrics.py
import pandas as pd
import numpy as np


# Änderungen für improved_benchmark_comparison in metrics.py

# Code-Block für verbesserte Benchmark-Vergleiche:
def improved_benchmark_comparison(returns, benchmark_returns):
    """
    Führt einen verbesserten Vergleich mit dem Benchmark durch.

    Parameters:
    -----------
    returns : pd.Series
        Tägliche Renditen der Strategie
    benchmark_returns : pd.Series
        Tägliche Renditen des Benchmarks

    Returns:
    --------
    dict
        Dictionary mit Alpha, Beta und anderen Benchmark-Vergleichsmetriken
    """
    import pandas as pd
    import numpy as np

    # Ergebniscontainer
    result = {'alpha': None, 'beta': None, 'tracking_error': None, 'information_ratio': None}

    # Frühe Rückkehr bei fehlenden Daten
    if returns is None or benchmark_returns is None:
        return result

    # Synchronisierung der Daten
    # Entferne NaN-Werte aus beiden Serien
    returns = returns.dropna()
    benchmark_returns = benchmark_returns.dropna()

    # Beschränke auf gemeinsame Indizes
    common_idx = returns.index.intersection(benchmark_returns.index)

    # Prüfe, ob wir genügend gemeinsame Datenpunkte haben
    if len(common_idx) < 2:
        print(f"Warnung: Zu wenige gemeinsame Datenpunkte für Benchmark-Vergleich ({len(common_idx)} gefunden)")
        return result

    returns_aligned = returns.loc[common_idx]
    benchmark_aligned = benchmark_returns.loc[common_idx]

    # Stelle sicher, dass beide Arrays die gleiche Form haben
    if len(returns_aligned) != len(benchmark_aligned):
        min_len = min(len(returns_aligned), len(benchmark_aligned))
        returns_aligned = returns_aligned.iloc[:min_len]
        benchmark_aligned = benchmark_aligned.iloc[:min_len]

    # Prüfe, ob wir genügend Datenpunkte haben
    if len(returns_aligned) < 2:
        print("Warnung: Zu wenige Datenpunkte für Benchmark-Vergleich nach Alignierung")
        return result

    try:
        # Umwandlung in NumPy-Arrays für effizientere Berechnung
        returns_array = np.array(returns_aligned)
        benchmark_array = np.array(benchmark_aligned)

        # Prüfe auf korrekte Dimensionen
        if returns_array.ndim > 1 and returns_array.shape[1] > 1:
            returns_array = returns_array[:, 0]

        if benchmark_array.ndim > 1 and benchmark_array.shape[1] > 1:
            benchmark_array = benchmark_array[:, 0]

        # Explizite Umformung zu 1D-Arrays
        returns_array = returns_array.flatten()
        benchmark_array = benchmark_array.flatten()

        # Prüfe, ob beide Arrays die gleiche Länge haben
        if len(returns_array) != len(benchmark_array):
            print(
                f"Warnung: Unterschiedliche Array-Längen: returns={len(returns_array)}, benchmark={len(benchmark_array)}")
            min_len = min(len(returns_array), len(benchmark_array))
            returns_array = returns_array[:min_len]
            benchmark_array = benchmark_array[:min_len]

        # Beta-Berechnung (Kovarianz / Varianz)
        # Manuell berechnen, um mehr Kontrolle zu haben
        mean_returns = np.mean(returns_array)
        mean_benchmark = np.mean(benchmark_array)

        # Kovarianz manuell berechnen
        covariance = np.mean((returns_array - mean_returns) * (benchmark_array - mean_benchmark))
        # Varianz des Benchmarks
        benchmark_variance = np.mean((benchmark_array - mean_benchmark) ** 2)

        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
        else:
            beta = 0

        # Alpha-Berechnung (annualisiert)
        rfr = 0  # Risikofreier Zinssatz (vereinfacht)

        # Berechne annualisierte Renditen
        annualized_return = (1 + np.mean(returns_array)) ** 252 - 1
        annualized_benchmark = (1 + np.mean(benchmark_array)) ** 252 - 1

        alpha = annualized_return - rfr - beta * (annualized_benchmark - rfr)

        # Tracking Error (Standardabweichung der Überrendite)
        excess_returns = returns_array - benchmark_array
        tracking_error = np.std(excess_returns) * np.sqrt(252)  # Annualisiert

        # Information Ratio
        excess_return_annualized = annualized_return - annualized_benchmark
        information_ratio = excess_return_annualized / tracking_error if tracking_error > 0 else 0

        # Aktualisiere Ergebnisse
        result['alpha'] = alpha
        result['beta'] = beta
        result['tracking_error'] = tracking_error
        result['information_ratio'] = information_ratio

        return result

    except Exception as e:
        print(f"Fehler bei der erweiterten Benchmark-Berechnung: {e}")
        import traceback
        traceback.print_exc()
        return result


def calculate_performance_metrics(returns, benchmark_returns=None):
    """
    Berechnet umfassende Performance-Metriken für eine Strategie.

    Parameters:
    -----------
    returns : pd.Series
        Tägliche Renditen der Strategie
    benchmark_returns : pd.Series
        Tägliche Renditen des Benchmarks

    Returns:
    --------
    dict
        Dictionary mit Performance-Metriken
    """
    # Entferne NaN-Werte
    returns = returns.dropna()

    # Berechne kumulative Renditen
    cum_returns = (1 + returns).cumprod()

    # Berechne Drawdowns
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak

    # Grundlegende Metriken
    total_return = cum_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)

    # Sharpe und Sortino Ratio
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0

    # Calmar Ratio (annualized return / max drawdown)
    max_drawdown = drawdown.min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Win/Loss-Statistiken
    winning_days = (returns > 0).sum()
    losing_days = (returns < 0).sum()
    win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

    # Durchschnittlicher Gewinn/Verlust
    avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0

    # Profit-Faktor
    profit_factor = (returns[returns > 0].sum() / abs(returns[returns < 0].sum())) if abs(
        returns[returns < 0].sum()) > 0 else float('inf')

    # Maximum Consecutive Wins/Losses
    returns_binary = returns.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    pos_streaks = (returns_binary > 0).astype(int)
    neg_streaks = (returns_binary < 0).astype(int)

    # Hilfsfunktion für Streak-Zählung
    def count_streaks(binary_series):
        streak_counts = []
        current_streak = 0

        for val in binary_series:
            if val == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streak_counts.append(current_streak)
                    current_streak = 0

        if current_streak > 0:
            streak_counts.append(current_streak)

        return max(streak_counts) if streak_counts else 0

    max_consecutive_wins = count_streaks(pos_streaks)
    max_consecutive_losses = count_streaks(neg_streaks)

    # Vergleich mit Benchmark, falls vorhanden
    alpha = None
    beta = None
    if benchmark_returns is not None:
        try:
            # Überprüfe, ob benchmark_returns eine Series ist
            if not isinstance(benchmark_returns, pd.Series):
                print("Warnung: benchmark_returns ist keine Pandas-Series. Konvertiere...")
                if isinstance(benchmark_returns, pd.DataFrame):
                    benchmark_returns = benchmark_returns.iloc[:, 0]
                else:
                    benchmark_returns = pd.Series(benchmark_returns)

            # Überprüfe, ob returns eine Series ist
            if not isinstance(returns, pd.Series):
                print("Warnung: returns ist keine Pandas-Series. Konvertiere...")
                if isinstance(returns, pd.DataFrame):
                    returns = returns.iloc[:, 0]
                else:
                    returns = pd.Series(returns)

            # Stelle sicher, dass die Indizes als DatetimeIndex vorliegen
            if not isinstance(returns.index, pd.DatetimeIndex):
                print("Warnung: returns Index ist kein DatetimeIndex.")
                # Versuche zu konvertieren, falls möglich
                if hasattr(returns.index, 'to_datetime'):
                    returns.index = pd.to_datetime(returns.index)

            if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
                print("Warnung: benchmark_returns Index ist kein DatetimeIndex.")
                # Versuche zu konvertieren, falls möglich
                if hasattr(benchmark_returns.index, 'to_datetime'):
                    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

            # Beschränke auf gemeinsame Indizes
            common_idx = returns.index.intersection(benchmark_returns.index)
            print(f"Gemeinsame Indizes: {len(common_idx)}")

            if len(common_idx) > 0:
                returns_aligned = returns.loc[common_idx]
                benchmark_aligned = benchmark_returns.loc[common_idx]

                # Überprüfe, ob genügend gemeinsame Datenpunkte vorhanden sind
                if len(returns_aligned) > 1 and len(benchmark_aligned) > 1:
                    # Stelle sicher, dass beide Arrays die gleiche Länge haben
                    if len(returns_aligned) != len(benchmark_aligned):
                        print(
                            f"Warnung: Unterschiedliche Längen nach Alignierung: returns={len(returns_aligned)}, benchmark={len(benchmark_aligned)}")
                        min_len = min(len(returns_aligned), len(benchmark_aligned))
                        returns_aligned = returns_aligned.iloc[:min_len]
                        benchmark_aligned = benchmark_aligned.iloc[:min_len]

                    # Beta berechnen
                    # Konvertiere zu NumPy Arrays für einfachere Berechnung
                    returns_array = returns_aligned.values
                    benchmark_array = benchmark_aligned.values

                    # Falls die Arrays noch mehrdimensional sind, flache sie ab
                    if returns_array.ndim > 1:
                        returns_array = returns_array.flatten()
                    if benchmark_array.ndim > 1:
                        benchmark_array = benchmark_array.flatten()

                    # Manually calculate covariance and variance
                    mean_returns = np.mean(returns_array)
                    mean_benchmark = np.mean(benchmark_array)

                    covariance = np.mean((returns_array - mean_returns) * (benchmark_array - mean_benchmark))
                    benchmark_variance = np.mean((benchmark_array - mean_benchmark) ** 2)

                    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0

                    # Alpha berechnen (annualisiert)
                    rfr = 0  # Risikofreier Zinssatz (vereinfacht)
                    benchmark_ret = (1 + np.mean(benchmark_array)) ** 252 - 1
                    alpha = annualized_return - rfr - beta * (benchmark_ret - rfr)

                    print(f"Alpha: {alpha}, Beta: {beta}")
                else:
                    print(
                        f"Zu wenige gemeinsame Datenpunkte für Alpha/Beta: returns={len(returns_aligned)}, benchmark={len(benchmark_aligned)}")
            else:
                print("Keine gemeinsamen Indizes für Alpha/Beta-Berechnung gefunden")
        except Exception as e:
            print(f"Fehler bei der Alpha/Beta-Berechnung: {e}")
            alpha = beta = None

    # Alle Metriken sammeln
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'alpha': alpha,
        'beta': beta
    }

    return metrics