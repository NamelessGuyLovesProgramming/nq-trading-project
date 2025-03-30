import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from src.visualization.charts import ChartVisualizer
from src.custom_html_writer import write_html_with_custom_interaction
import tempfile
import os
import pandas as pd
import numpy as np


def downsample_data(df, max_points=10000):
    """
    Intelligently downsample dataframe for visualization while preserving important patterns.

    Args:
        df: DataFrame with time-series data
        max_points: Maximum number of data points to return

    Returns:
        Downsampled DataFrame
    """
    # If the dataframe is already small enough, return it as is
    if len(df) <= max_points:
        return df

    # Ensure necessary OHLCV columns are numeric
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in ohlcv_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"Warnung: Konnte Spalte {col} nicht in numerischen Typ konvertieren")

    # Calculate sampling factor
    sampling_factor = len(df) // max_points

    # For OHLCV data, we want to preserve price movement patterns
    # This custom resampling preserves highs, lows and price movements better than simple sampling
    result = []

    try:
        for i in range(0, len(df), sampling_factor):
            chunk = df.iloc[i:i + sampling_factor]
            if len(chunk) > 0:
                # For each chunk, keep the first, highest, lowest, and last points
                points_to_keep = []

                # First point
                points_to_keep.append(chunk.iloc[0])

                # Highest high (if High column exists and is numeric)
                if 'High' in chunk.columns and pd.api.types.is_numeric_dtype(chunk['High']):
                    high_idx = chunk['High'].idxmax()
                    if high_idx not in [p.name for p in points_to_keep]:
                        points_to_keep.append(chunk.loc[high_idx])

                # Lowest low (if Low column exists and is numeric)
                if 'Low' in chunk.columns and pd.api.types.is_numeric_dtype(chunk['Low']):
                    low_idx = chunk['Low'].idxmin()
                    if low_idx not in [p.name for p in points_to_keep]:
                        points_to_keep.append(chunk.loc[low_idx])

                # Last point (if different from the above)
                if len(chunk) > 1 and chunk.iloc[-1].name not in [p.name for p in points_to_keep]:
                    points_to_keep.append(chunk.iloc[-1])

                # Sort by index to maintain time order
                points_to_keep.sort(key=lambda x: x.name)

                # Add to result
                result.extend(points_to_keep)
    except Exception as e:
        print(f"Fehler beim Downsampling: {e}")
        # Fallback: Einfaches Downsampling
        return df.iloc[::sampling_factor].copy()

    # Convert back to DataFrame
    try:
        downsampled_df = pd.DataFrame(result)

        # Apply additional smoothing if there are indicators (nur für numerische Spalten)
        indicator_cols = [col for col in downsampled_df.columns
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Signal']]

        for col in indicator_cols:
            if col in downsampled_df.columns:
                # Prüfe, ob die Spalte numerisch ist, bevor EMA angewendet wird
                if pd.api.types.is_numeric_dtype(downsampled_df[col]):
                    # Use EMA to smooth the indicators after downsampling
                    if not downsampled_df[col].isna().all():  # Skip if all values are NaN
                        try:
                            downsampled_df[col] = downsampled_df[col].ewm(span=5).mean()
                        except Exception as e:
                            print(f"Fehler bei der Glättung von {col}: {e}")

        print(f"Downsampling abgeschlossen: {len(df)} auf {len(downsampled_df)} Datenpunkte reduziert")
        return downsampled_df
    except Exception as e:
        print(f"Fehler bei der DataFrame-Erstellung: {e}")
        # Fallback bei Fehler
        return df.iloc[::sampling_factor].copy()


def render_interactive_chart(fig, height=600):
    """
    Rendert einen Plotly-Chart mit den benutzerdefinierten Interaktionen aus custom_html_writer.py

    Args:
        fig: Plotly Figure-Objekt
        height: Höhe des Charts in Pixeln
    """
    # Only adjust the bottom margin to make space for x-axis without changing other settings
    current_margin = fig.layout.margin or {}
    current_margin_dict = {}
    if hasattr(current_margin, 'l'):
        current_margin_dict['l'] = current_margin.l
    if hasattr(current_margin, 'r'):
        current_margin_dict['r'] = current_margin.r
    if hasattr(current_margin, 't'):
        current_margin_dict['t'] = current_margin.t
    if hasattr(current_margin, 'b'):
        current_margin_dict['b'] = current_margin.b

    # Set defaults if values are missing
    if 'l' not in current_margin_dict:
        current_margin_dict['l'] = 50
    if 'r' not in current_margin_dict:
        current_margin_dict['r'] = 60
    if 't' not in current_margin_dict:
        current_margin_dict['t'] = 80
    if 'b' not in current_margin_dict:
        current_margin_dict['b'] = 50

    # Just increase bottom margin without changing other margins
    current_margin_dict['b'] = max(70, current_margin_dict['b'])

    # Create a new x-axis dict instead of trying to convert the XAxis object
    xaxis_dict = {
        'visible': True,
        'showticklabels': True
    }

    fig.update_layout(
        margin=current_margin_dict
    )

    fig.update_xaxes(
        visible=True,
        showticklabels=True,
        title_standoff=25  # Increase standoff for title
    )

    # Rest of the function remains the same
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as f:
        temp_path = f.name
        write_html_with_custom_interaction(fig, temp_path)

    with open(temp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Add minimal JavaScript to ensure x-axis visibility without affecting other customizations
    streamlit_js = """
    <script>
        // Existing observers and event handlers
        const resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                if (entry.target.classList.contains('js-plotly-plot')) {
                    window.dispatchEvent(new Event('resize'));
                }
            }
        });

        document.querySelectorAll('.js-plotly-plot').forEach(plot => {
            resizeObserver.observe(plot);
        });

        window.addEventListener('wheel', function(e) {
            e.stopPropagation();
        }, true);

        // Focused fix just for x-axis visibility
        window.addEventListener('load', function() {
            setTimeout(function() {
                var gd = document.querySelector('.js-plotly-plot');
                if (window.Plotly && gd) {
                    // Only update x-axis visibility without changing other settings
                    Plotly.relayout(gd, {
                        'xaxis.visible': true,
                        'xaxis.showticklabels': true
                    });
                }
            }, 100); // Small delay to ensure chart is fully loaded
        });
    </script>
    """

    # Focused CSS fix for x-axis visibility
    streamlit_css = """
    <style>
    /* Target only the x-axis elements to ensure visibility */
    .js-plotly-plot .plotly .main-svg .xaxislayer-above {
        visibility: visible !important;
    }
    .js-plotly-plot .plotly .main-svg .xaxislayer-above .xtick text {
        visibility: visible !important;
        opacity: 1 !important;
    }
    /* Add a bit of padding at the bottom to ensure axis is visible */
    .js-plotly-plot {
        padding-bottom: 10px;
    }
    </style>
    """

    html_content = html_content.replace('</body>', streamlit_js + streamlit_css + '</body>')

    os.unlink(temp_path)

    # Add a little extra height just for the axis
    components.html(html_content, height=height + 20)


def filter_weekend_days(data):
    """
    Filtert Wochenendtage und Tage ohne Daten aus dem DataFrame.

    Args:
        data: DataFrame mit Zeitindex

    Returns:
        DataFrame ohne Wochenendtage und Tage ohne Daten
    """
    # Sicherstellen, dass der Index ein DatetimeIndex ist
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except:
            print("Konnte Index nicht in DatetimeIndex konvertieren")
            return data

    # Wochenendtage filtern (5=Samstag, 6=Sonntag)
    weekday_data = data[data.index.dayofweek < 5].copy()

    # Zusätzlich Tage filtern, an denen keine Daten vorhanden sind
    # (z.B. Feiertage oder Handelspausen)
    if 'Volume' in weekday_data.columns:
        # Nur filtern, wenn Volume-Spalte vorhanden und numerisch ist
        if pd.api.types.is_numeric_dtype(weekday_data['Volume']):
            non_zero_volume = weekday_data['Volume'] > 0
            if non_zero_volume.any():
                weekday_data = weekday_data[non_zero_volume]

    # Auch prüfen, ob alle OHLC-Werte gleich sind (kein Handel)
    required_cols = ['Open', 'High', 'Low', 'Close']
    if all(col in weekday_data.columns for col in required_cols):
        # Nur filtern, wenn alle Spalten vorhanden und numerisch sind
        if all(pd.api.types.is_numeric_dtype(weekday_data[col]) for col in required_cols):
            # Finde Tage, an denen sich der Preis verändert hat (High != Low)
            price_moved = weekday_data['High'] != weekday_data['Low']
            if price_moved.any():
                weekday_data = weekday_data[price_moved]

    print(
        f"Wochenendtage und handelsfreie Tage gefiltert: {len(data) - len(weekday_data)} von {len(data)} Tagen entfernt")
    return weekday_data


def create_candlestick_chart(data, indicators=None, signals=None, height=600, date_range=None, max_points=10000,
                             skip_weekends=True):
    """
    Erstellt und rendert einen Candlestick-Chart mit den benutzerdefinierten Interaktionen

    Args:
        data: DataFrame mit OHLCV-Daten
        indicators: Liste von Indikatoren, die angezeigt werden sollen
        signals: DataFrame mit Signalen
        height: Höhe des Charts in Pixeln
        date_range: Tuple mit (start_date, end_date) für Zeitfilterung
        max_points: Maximale Anzahl von Datenpunkten für Downsampling
        skip_weekends: Wenn True, werden Wochenendtage übersprungen
    """
    # Apply date range filter if provided
    filtered_data = data.copy()
    if date_range is not None and date_range[0] is not None and date_range[1] is not None:
        start_date, end_date = date_range
        mask = (filtered_data.index >= start_date) & (filtered_data.index <= end_date)
        filtered_data = filtered_data.loc[mask]

        if signals is not None:
            signals = signals.loc[mask]

    # Filtere Wochenendtage, wenn gewünscht
    if skip_weekends:
        filtered_data = filter_weekend_days(filtered_data)
        if signals is not None:
            signals = signals[signals.index.isin(filtered_data.index)]

    # Estimate data size
    approx_size_mb = filtered_data.memory_usage(deep=True).sum() / (1024 * 1024)

    # Check if downsampling is needed
    if len(filtered_data) > max_points or approx_size_mb > 100:
        st.warning(
            f"Dataset is large ({len(filtered_data)} points, ~{approx_size_mb:.1f} MB). Downsampling for visualization.")
        filtered_data = downsample_data(filtered_data, max_points)

        # Also downsample signals if they exist
        if signals is not None:
            # Only keep signals for dates in the downsampled data
            downsampled_dates = set(filtered_data.index)
            signals = signals[signals.index.isin(downsampled_dates)]

    # Verwende die bestehende ChartVisualizer-Klasse
    visualizer = ChartVisualizer()
    chart = visualizer.plot_candlestick_with_indicators(
        filtered_data,
        indicators=indicators,
        signals=signals
    )

    # Konfiguriere Chart für das Überspringen von Tagen ohne Daten
    # WICHTIG: Wir fügen nur die rangebreaks-Option hinzu, ohne andere Einstellungen zu ändern
    if skip_weekends:
        chart.update_xaxes(
            rangebreaks=[
                dict(pattern='day of week', bounds=[5, 7])  # Wochenenden überspringen
            ]
        )

    # Rendere den Chart mit benutzerdefinierten Interaktionen
    render_interactive_chart(chart, height=height)


def create_backtest_chart(backtest_results, height=600, max_points=5000):
    """
    Erstellt und rendert einen Backtest-Chart mit den benutzerdefinierten Interaktionen

    Args:
        backtest_results: Dictionary mit Backtest-Ergebnissen
        height: Höhe des Charts in Pixeln
        max_points: Maximale Anzahl von Datenpunkten
    """
    # Check if we need to downsample the backtest results
    portfolio_values = backtest_results['portfolio_value']
    drawdowns = backtest_results['drawdown']

    if len(portfolio_values) > max_points:
        # Downsample while preserving trends
        sampling_factor = len(portfolio_values) // max_points + 1

        # For portfolio values, keep min/max within each chunk and high-volatility points
        downsampled_portfolio = []
        downsampled_drawdown = []

        for i in range(0, len(portfolio_values), sampling_factor):
            chunk_pv = portfolio_values.iloc[i:i + sampling_factor]
            chunk_dd = drawdowns.iloc[i:i + sampling_factor]

            if len(chunk_pv) > 0:
                # Always include first and last points
                downsampled_portfolio.append((chunk_pv.index[0], chunk_pv.iloc[0]))

                if len(chunk_pv) > 2:
                    # Find max and min in the chunk
                    max_idx = chunk_pv.idxmax()
                    min_idx = chunk_pv.idxmin()

                    # Add max and min if they're not the first or last point
                    if max_idx != chunk_pv.index[0] and max_idx != chunk_pv.index[-1]:
                        downsampled_portfolio.append((max_idx, chunk_pv.loc[max_idx]))
                    if min_idx != chunk_pv.index[0] and min_idx != chunk_pv.index[-1] and min_idx != max_idx:
                        downsampled_portfolio.append((min_idx, chunk_pv.loc[min_idx]))

                # Add the last point if the chunk has more than one point
                if len(chunk_pv) > 1 and chunk_pv.index[-1] != chunk_pv.index[0]:
                    downsampled_portfolio.append((chunk_pv.index[-1], chunk_pv.iloc[-1]))

                # Do the same for drawdown
                downsampled_drawdown.append((chunk_dd.index[0], chunk_dd.iloc[0]))

                if len(chunk_dd) > 2:
                    # For drawdown, we care most about the maximum drawdowns
                    max_dd_idx = chunk_dd.idxmin()  # Min because drawdowns are negative

                    # Add max drawdown if it's not already included
                    if max_dd_idx != chunk_dd.index[0] and max_dd_idx != chunk_dd.index[-1]:
                        downsampled_drawdown.append((max_dd_idx, chunk_dd.loc[max_dd_idx]))

                if len(chunk_dd) > 1 and chunk_dd.index[-1] != chunk_dd.index[0]:
                    downsampled_drawdown.append((chunk_dd.index[-1], chunk_dd.iloc[-1]))

        # Sort by index
        downsampled_portfolio.sort(key=lambda x: x[0])
        downsampled_drawdown.sort(key=lambda x: x[0])

        # Create Series from downsampled data
        downsampled_pv_index = [x[0] for x in downsampled_portfolio]
        downsampled_pv_values = [x[1] for x in downsampled_portfolio]
        downsampled_dd_index = [x[0] for x in downsampled_drawdown]
        downsampled_dd_values = [x[1] for x in downsampled_drawdown]

        # Create new Series
        portfolio_values = pd.Series(downsampled_pv_values, index=downsampled_pv_index)
        drawdowns = pd.Series(downsampled_dd_values, index=downsampled_dd_index)

        # Create a modified backtest_results dictionary
        modified_results = backtest_results.copy()
        modified_results['portfolio_value'] = portfolio_values
        modified_results['drawdown'] = drawdowns

        # Notify the user that downsampling was applied
        st.info(
            f"Backtest results were downsampled from {len(backtest_results['portfolio_value'])} to {len(portfolio_values)} points for visualization.")

        # Verwende die bestehende ChartVisualizer-Klasse mit modifizierten Daten
        visualizer = ChartVisualizer()
        chart = visualizer.plot_backtest_results(modified_results)
    else:
        # Verwende die bestehende ChartVisualizer-Klasse mit Originaldaten
        visualizer = ChartVisualizer()
        chart = visualizer.plot_backtest_results(backtest_results)

    # Rendere den Chart mit benutzerdefinierten Interaktionen
    render_interactive_chart(chart, height=height)