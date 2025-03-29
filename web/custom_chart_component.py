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