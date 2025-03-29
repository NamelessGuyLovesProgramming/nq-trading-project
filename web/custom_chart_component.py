import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from src.visualization.charts import ChartVisualizer
from src.custom_html_writer import write_html_with_custom_interaction
import tempfile
import os


def render_interactive_chart(fig, height=600):
    """
    Rendert einen Plotly-Chart mit den benutzerdefinierten Interaktionen aus custom_html_writer.py

    Args:
        fig: Plotly Figure-Objekt
        height: Höhe des Charts in Pixeln
    """
    # Erstelle temporäre Datei für das HTML
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as f:
        temp_path = f.name
        # Verwende die bestehende Funktion, um den Chart mit benutzerdefinierten Interaktionen zu erstellen
        write_html_with_custom_interaction(fig, temp_path)

    # Lese die erzeugte HTML-Datei
    with open(temp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Füge zusätzliches JavaScript für Streamlit-Integration hinzu
    streamlit_js = """
    <script>
        // Stellen sicher, dass das Chart im Streamlit-iFrame korrekt skaliert wird
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

        // Fix für Mausrad-Events im iFrame
        window.addEventListener('wheel', function(e) {
            e.stopPropagation();
        }, true);
    </script>
    """

    # Füge das Streamlit-spezifische JavaScript zum HTML-Content hinzu
    html_content = html_content.replace('</body>', streamlit_js + '</body>')

    # Lösche die temporäre Datei
    os.unlink(temp_path)

    # Render HTML mit den benutzerdefinierten Interaktionen
    components.html(html_content, height=height)

def create_candlestick_chart(data, indicators=None, signals=None, height=600):
    """
    Erstellt und rendert einen Candlestick-Chart mit den benutzerdefinierten Interaktionen

    Args:
        data: DataFrame mit OHLCV-Daten
        indicators: Liste von Indikatoren, die angezeigt werden sollen
        signals: DataFrame mit Signalen
        height: Höhe des Charts in Pixeln
    """
    # Verwende die bestehende ChartVisualizer-Klasse
    visualizer = ChartVisualizer()
    chart = visualizer.plot_candlestick_with_indicators(
        data,
        indicators=indicators,
        signals=signals
    )

    # Rendere den Chart mit benutzerdefinierten Interaktionen
    render_interactive_chart(chart, height=height)


def create_backtest_chart(backtest_results, height=600):
    """
    Erstellt und rendert einen Backtest-Chart mit den benutzerdefinierten Interaktionen

    Args:
        backtest_results: Dictionary mit Backtest-Ergebnissen
        height: Höhe des Charts in Pixeln
    """
    # Verwende die bestehende ChartVisualizer-Klasse
    visualizer = ChartVisualizer()
    chart = visualizer.plot_backtest_results(backtest_results)

    # Rendere den Chart mit benutzerdefinierten Interaktionen
    render_interactive_chart(chart, height=height)