import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class ChartVisualizer:
    def __init__(self):
        """
        Initialisiert den ChartVisualizer.
        """
        pass

    def _prepare_dataframe(self, df):
        """
        Bereitet das DataFrame für die Visualisierung vor, behandelt auch MultiIndex-Spalten.
        """
        # Kopie erstellen, um das Original nicht zu verändern
        df = df.copy()

        # Prüfen, ob wir MultiIndex-Spalten haben
        if isinstance(df.columns, pd.MultiIndex):
            print("MultiIndex-Spalten erkannt")

            # Extrahiere die ersten Level der Spalten
            ohlc_cols = {'Open': None, 'High': None, 'Low': None, 'Close': None, 'Volume': None}

            # Finde die richtigen Spalten basierend auf dem ersten Level
            for col in df.columns:
                if col[0] in ohlc_cols:
                    ohlc_cols[col[0]] = col

            print(f"Gefundene OHLCV-Spalten: {ohlc_cols}")

            # Erstelle ein neues DataFrame mit den benötigten Spalten
            new_df = pd.DataFrame(index=df.index)
            for simple_name, multi_name in ohlc_cols.items():
                if multi_name is not None:
                    new_df[simple_name] = df[multi_name]

            # Füge Indikatoren hinzu (ohne Symbol-Teil)
            for col in df.columns:
                if col[0] not in ohlc_cols and col[0] != '':
                    new_df[col[0]] = df[col]

            return new_df
        else:
            # Kein MultiIndex, überprüfe trotzdem auf lowercase-Varianten
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_columns = df.columns.tolist()

            # Prüfen auf lowercase-Varianten und umbenennen
            rename_dict = {}
            for col in required_cols:
                if col not in df_columns and col.lower() in df_columns:
                    rename_dict[col.lower()] = col

            if rename_dict:
                df = df.rename(columns=rename_dict)
                print(f"Spalten umbenannt: {rename_dict}")

            return df

    def plot_candlestick(self, df, title='NQ Candlestick Chart'):
        """
        Erstellt ein Candlestick-Chart im TradingView-Stil.

        - Standardansicht: Halber Tag bis zur letzten Kerze
        - Preisanzeige rechts
        - Fadenkreuz als Standardwerkzeug
        - Automatische Relation zwischen X und Y-Achsen

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame mit OHLCV-Daten
        title : str
            Titel des Charts

        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly-Figure-Objekt im TradingView-Stil
        """
        # DataFrame vorbereiten
        df = self._prepare_dataframe(df)

        # Debug-Ausgabe
        print(f"DataFrame-Index-Typ: {type(df.index)}")
        print(f"DataFrame-Index-Beispiel: {df.index[:5]}")
        print(f"DataFrame-Spalten nach Vorbereitung: {df.columns.tolist()}")

        # Stelle sicher, dass der Index ein DatetimeIndex ist
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                print("Index zu DatetimeIndex konvertiert")
            except Exception as e:
                print(f"Fehler beim Konvertieren des Index: {e}")

        # Entferne NaN-Werte in den OHLC-Spalten
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        # Debug: Daten-Check
        print(f"Erste 5 Zeilen des DataFrame nach Vorbereitung:")
        print(df[['Open', 'High', 'Low', 'Close']].head())

        # Erstelle Figure
        fig = go.Figure()

        # Füge Candlesticks hinzu
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))

        # Berechne Standardansicht (halber Tag bis letzte Kerze)
        end_date = df.index[-1]
        start_date = end_date - pd.Timedelta(hours=12)

        fig.update_layout(
            title=title,
            xaxis_title='Datum',
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            dragmode='zoom',  # Ändert den Standard-Modus von 'pan' zu 'zoom'
            modebar_add=['drawline', 'eraseshape'],
            newshape=dict(line_color='blue'),
            # Automatisches Verhältnis zwischen X und Y Achsen
            autosize=True,
            margin=dict(l=50, r=60, b=50, t=80),
            # Hintergrundfarbe für TradingView-Look
            plot_bgcolor='rgb(250, 250, 250)',
            # Aktiviere Fadenkreuz als Standard
            hovermode='x unified'
        )

        # Y-Achse auf rechter Seite
        fig.update_yaxes(
            title_text='Preis',
            side='right',
            showgrid=True,
            zeroline=False,
            automargin=True,
            ticklabelposition='outside right',
            # Erweiterte Konfiguration für Y-Achse: Ziehen erlauben
            fixedrange=False,
            scaleanchor='x',
            constrain='domain'
        )

        # X-Achse konfigurieren
        fig.update_xaxes(
            range=[start_date, end_date],  # Standardansicht: halber Tag
            showspikes=True,  # Spike-Linien für Fadenkreuz
            spikemode='across',
            spikesnap='cursor',
            showline=True,
            showgrid=True,
            # Erweiterte Konfiguration für X-Achse: Ziehen erlauben
            fixedrange=False,
            constrain='domain',
            # Zoom-Konfiguration
            rangeslider_visible=False
        )

        # Konfiguration für Mausinteraktionen hinzufügen
        fig.update_layout(
            # Aktiviere erweiterte Zoom/Pan-Funktionen
            dragmode='zoom',
            # Konfiguriere Verhalten beim Zoomen
            clickmode='event+select',
            # Spezielle Reaktion auf Mausrad-Event
            # Leider kann das Mausrad-Verhalten nicht direkt in Plotly konfiguriert werden,
            # daher benötigen wir JavaScript für die individuellen Anforderungen
        )

        # Füge Custom-JavaScript für erweiterte Interaktionen hinzu
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Zoom Reset",
                            method="relayout",
                            args=["xaxis.range", [None, None]]
                        )
                    ],
                    x=0.05,
                    y=0.99,
                    xanchor="left",
                    yanchor="top"
                )
            ]
        )

        # Bei Nutzung in HTML-Output, schreibe zusätzlichen HTML-Code
        # Der folgende Code wird erst beim Schreiben der HTML-Datei wirksam
        html_post_script = """
        <script>
        // Optimierte Scrollzoom-Funktion nach dem Laden
        window.addEventListener('load', function() {
            var gd = document.querySelector('.plotly-graph-div');
            if (!gd) return;

            // Erweiterte Mausrad-Funktion
            gd.addEventListener('wheel', function(e) {
                // Nur fortfahren, wenn wir über dem Hauptchart sind
                if (e.target.tagName === 'rect' || 
                    e.target.tagName === 'svg' || 
                    e.target.classList.contains('main-svg')) {

                    e.preventDefault();
                    var xaxis = gd._fullLayout.xaxis;

                    // Bestimme Zoom-Richtung
                    var direction = e.deltaY < 0 ? -1 : 1;
                    var ctrlKey = e.ctrlKey || e.metaKey;

                    // Berechne Mausposition relative zum Chart
                    var rect = gd.getBoundingClientRect();
                    var x = e.clientX - rect.left;
                    var width = rect.width;
                    var xpercent = x / width;

                    // Aktuelles Bereich
                    var xmin = xaxis.range[0];
                    var xmax = xaxis.range[1];
                    var xrange = xmax - xmin;

                    // Spezielle Zoomfaktoren
                    var factor = ctrlKey ? 0.15 : 0.05;
                    var zoom = 1 + factor * direction;

                    // Neue Bereiche berechnen
                    var newRange;
                    if (ctrlKey) {
                        // STRG+Mausrad: Beidseitiges Zoom vom Cursor aus
                        var xcenter = xmin + xrange * xpercent;
                        var newXMin = xcenter - (xcenter - xmin) * zoom;
                        var newXMax = xcenter + (xmax - xcenter) * zoom;
                        newRange = [newXMin, newXMax];
                    } else {
                        // Standard Mausrad: Strecken/Stauchen in eine Richtung
                        var newXMin = xmin - (direction < 0 ? xrange * factor : 0);
                        var newXMax = xmax + (direction > 0 ? xrange * factor : 0);
                        newRange = [newXMin, newXMax];
                    }

                    // Layout aktualisieren
                    Plotly.relayout(gd, {
                        'xaxis.range': newRange
                    });
                }
            }, {passive: false});

            // Interaktion mit den Achsen verbessern
            function setupAxisDrag(axisName) {
                var axisElements = document.querySelectorAll('.' + axisName + 'tick');
                axisElements.forEach(function(el) {
                    el.style.cursor = 'ew-resize';
                    el.addEventListener('mousedown', function(e) {
                        var startX = e.clientX;
                        var axis = gd._fullLayout[axisName];
                        var startRange = [...axis.range];
                        var startWidth = startRange[1] - startRange[0];

                        function mousemove(e) {
                            var dx = (e.clientX - startX) / gd.clientWidth;
                            var scale = e.shiftKey ? 5 : 1;
                            var newRange = [
                                startRange[0] - dx * startWidth * scale,
                                startRange[1] - dx * startWidth * scale
                            ];
                            var update = {};
                            update[axisName + '.range'] = newRange;
                            Plotly.relayout(gd, update);
                        }

                        function mouseup() {
                            document.removeEventListener('mousemove', mousemove);
                            document.removeEventListener('mouseup', mouseup);
                        }

                        document.addEventListener('mousemove', mousemove);
                        document.addEventListener('mouseup', mouseup);
                        e.preventDefault();
                    });
                });
            }

            // Achseninteraktion aktivieren
            setupAxisDrag('xaxis');

            // Zusätzliche Y-Achsen-Interaktionen, falls benötigt
            var yaxisElements = document.querySelectorAll('.ytick');
            yaxisElements.forEach(function(el) {
                el.style.cursor = 'ns-resize';
            });
        });
        </script>
        """

        # Diese Methode gibt das Plotly-Figure-Objekt zurück
        # Das HTML-Post-Script wird beim Schreiben der HTML-Datei angehängt
        # Speichere die Zusatzskripte in einer Eigenschaft
        fig._user_html_post = html_post_script

        return fig

    # Änderungen für die Methode plot_candlestick_with_indicators in src/visualization/charts.py

    # Änderungen für die Methode plot_candlestick_with_indicators in src/visualization/charts.py

    # Änderung für die plot_candlestick_with_indicators Methode in src/visualization/charts.py

    def plot_candlestick_with_indicators(self, df, indicators=None, signals=None, skip_weekends=True):
        """
        Erstellt ein Candlestick-Chart mit Indikatoren und Signalen im TradingView-Stil.

        - Standardansicht: Alle verfügbaren Candles anzeigen
        - Preisanzeige und Y-Achse rechts
        - Bessere Proportionen der Kerzenkörper
        - Option zum Überspringen von Wochenendtagen
        """
        # DataFrame vorbereiten
        df = self._prepare_dataframe(df)

        # Debug-Ausgabe
        print(f"DataFrame-Index-Typ: {type(df.index)}")
        print(f"DataFrame-Index-Beispiel: {df.index[:5]}")
        print(f"DataFrame-Spalten nach Vorbereitung: {df.columns.tolist()}")
        print(f"Anzahl der Datenpunkte: {len(df)}")

        # Stelle sicher, dass der Index ein DatetimeIndex ist
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
                print("Index zu DatetimeIndex konvertiert")
            except Exception as e:
                print(f"Fehler beim Konvertieren des Index: {e}")

        # Entferne NaN-Werte in den OHLC-Spalten
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        # Berechne Range der Preise für optimale Skalierung
        price_range = df['High'].max() - df['Low'].min()
        price_padding = price_range * 0.05  # 5% Polsterung oben und unten
        y_min = df['Low'].min() - price_padding
        y_max = df['High'].max() + price_padding

        # Erstelle Subplot für Hauptchart und Volumen
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02,
                            row_heights=[0.7, 0.3],
                            subplot_titles=('Preis', 'Volumen'))

        # Füge Candlesticks hinzu mit verbesserten Darstellungsoptionen
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks',
            increasing_line_color='green',
            decreasing_line_color='red',
            increasing_fillcolor='green',  # Füllung für steigende Kerzen
            decreasing_fillcolor='red',  # Füllung für fallende Kerzen
            line=dict(width=1),  # Dünnere Linie für bessere Darstellung
            showlegend=False,
            # Verbessere die Darstellung der Kerzenkörper
            whiskerwidth=0.2,  # Schmälere Dochte
            opacity=1  # Vollständige Deckkraft
        ), row=1, col=1)

        # Füge Indikatoren hinzu, falls gewünscht
        if indicators:
            colors = ['rgba(13, 71, 161, 0.9)', 'rgba(140, 20, 252, 0.9)',
                      'rgba(20, 180, 30, 0.9)', 'rgba(180, 30, 20, 0.9)']

            for i, indicator in enumerate(indicators):
                if indicator in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[indicator],
                        name=indicator,
                        line=dict(color=colors[i % len(colors)], width=1.5),
                        visible='legendonly'
                    ), row=1, col=1)

        # Füge Volumen hinzu
        if 'Volume' in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker=dict(color='rgba(68, 68, 68, 0.7)'),
                showlegend=True,
                visible='legendonly'
            ), row=2, col=1)
        else:
            print("Warnung: Volume-Spalte nicht gefunden")

        # Füge Signale hinzu, wenn vorhanden
        if signals is not None:
            # Falls signals ebenfalls MultiIndex-Spalten hat, vorbereiten
            if isinstance(signals.columns, pd.MultiIndex):
                signals = self._prepare_dataframe(signals)

            # Prüfe, ob 'Signal' im DataFrame vorhanden ist
            if 'Signal' in signals.columns:
                buy_signals = signals[signals['Signal'] == 1]
                sell_signals = signals[signals['Signal'] == -1]

                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Close'],
                        mode='markers',
                        name='Kaufsignal',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='green',
                            line=dict(width=2, color='darkgreen')
                        ),
                        visible='legendonly'
                    ), row=1, col=1)

                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['Close'],
                        mode='markers',
                        name='Verkaufssignal',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='red',
                            line=dict(width=2, color='darkred')
                        ),
                        visible='legendonly'
                    ), row=1, col=1)
            else:
                print("Warnung: 'Signal' Spalte nicht in Signals-DataFrame gefunden.")

        # Optimiere die Anzeige basierend auf der Datenmenge
        end_date = df.index[-1]

        # Bei sehr vielen Datenpunkten, zeige die letzten X Kerzen standardmäßig
        if len(df) > 200:
            # Zeige etwa die letzten 100-150 Kerzen standardmäßig
            start_date = df.index[max(0, len(df) - 150)]
        else:
            # Bei weniger Datenpunkten, zeige alle
            start_date = df.index[0]

        # Layout-Einstellungen
        fig.update_layout(
            title='Candlestick Chart',
            xaxis_title='Datum',
            height=800,
            width=1200,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # TradingView-ähnliche Einstellungen
            dragmode='zoom',
            modebar_add=['drawline', 'eraseshape', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
            autosize=True,
            margin=dict(l=50, r=60, b=50, t=80),
            plot_bgcolor='rgb(250, 250, 250)',
            hovermode='x unified',
            # Reset-Button hinzufügen
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Reset Zoom",
                            method="relayout",
                            args=[{
                                "xaxis.range": [start_date, end_date],
                                "yaxis.range": [y_min, y_max]
                            }]
                        )
                    ],
                    direction="left",
                    pad={"r": 10, "t": 10},
                    x=0.05,
                    y=1.05,
                    xanchor="left",
                    yanchor="top"
                )
            ]
        )

        # Konfiguriere X-Achse für das Überspringen von Tagen ohne Daten (Wochenenden)
        if skip_weekends:
            fig.update_xaxes(
                type='category',  # Verwende kategorische Achse statt kontinuierlicher Zeitachse
                rangeslider_visible=False,  # Rangeslider ausblenden
                rangebreaks=[
                    # Wochenenden überspringen
                    dict(pattern='day of week', bounds=[5, 7])
                ]
            )
        else:
            # Standard X-Achse konfigurieren
            fig.update_xaxes(
                rangeslider_visible=False,  # Rangeslider ausblenden
                automargin=True,
                range=[start_date, end_date],  # Setze Standardzoom
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                showline=True,
                showgrid=True,
                fixedrange=False,
                constrain='domain'
            )

        # Y-Achse konfigurieren mit festen Grenzen für bessere Proportionen der Candlesticks
        fig.update_yaxes(
            title_text='Preis',
            row=1,
            col=1,
            side='right',
            showgrid=True,
            zeroline=False,
            automargin=True,
            ticklabelposition='outside right',
            fixedrange=False,
            constrain='domain',
            range=[y_min, y_max]  # Setze feste Range für bessere Proportionen
        )

        # Volumen-Y-Achse
        fig.update_yaxes(
            title_text='Volumen',
            row=2,
            col=1,
            side='right',
            showgrid=True,
            zeroline=False,
            automargin=True,
            ticklabelposition='outside right',
            fixedrange=False,
            constrain='domain'
        )

        # Weitere Anpassungen für beste Darstellung der Candlesticks
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            showlegend=True,
            # Optimiere Darstellung der Candlesticks
            yaxis=dict(
                autorange=False,  # Deaktiviere Autorange für bessere Kontrolle
                range=[y_min, y_max],  # Verwende berechneten Bereich
                tickmode='auto',
                nticks=10  # Angemessene Anzahl von Ticks für bessere Lesbarkeit
            )
        )

        return fig
    def plot_backtest_results(self, backtest_results, benchmark=None):
        """
        Visualisiert Backtest-Ergebnisse im TradingView-Stil.

        - Preisanzeige rechts
        - Fadenkreuz als Standardwerkzeug
        - Automatische Relation zwischen X und Y-Achsen
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03,
                            subplot_titles=('Portfolio-Wert', 'Drawdown'))

        # Portfolio-Wert
        fig.add_trace(go.Scatter(
            x=backtest_results['portfolio_value'].index,
            y=backtest_results['portfolio_value'].values,
            name='Strategie',
            line=dict(color='rgba(0, 100, 0, 0.8)', width=2)
        ), row=1, col=1)

        # Benchmark, falls vorhanden
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                name='Benchmark',
                line=dict(color='rgba(100, 100, 100, 0.8)', width=1.5)
            ), row=1, col=1)

        # Drawdown
        fig.add_trace(go.Scatter(
            x=backtest_results['drawdown'].index,
            y=backtest_results['drawdown'].values,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='rgba(200, 0, 0, 0.7)')
        ), row=2, col=1)

        # Berechne Standardansicht (letzten Monat zeigen)
        end_date = backtest_results['portfolio_value'].index[-1]
        start_date = end_date - pd.Timedelta(days=30)

        # Layout im TradingView-Stil mit erweiterten Interaktionen
        fig.update_layout(
            title='Backtest-Ergebnisse',
            xaxis_title='Datum',
            height=800,
            width=1200,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # TradingView-ähnliche Einstellungen mit Interaktivität
            dragmode='zoom',  # Ändern von 'pan' zu 'zoom' für bessere Kontrolle
            modebar_add=['drawline', 'eraseshape', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
            autosize=True,
            margin=dict(l=50, r=60, b=50, t=80),
            plot_bgcolor='rgb(250, 250, 250)',
            hovermode='x unified'  # Fadenkreuz als Standard
        )

        # Y-Achsen auf der rechten Seite mit erweiterter Interaktivität
        fig.update_yaxes(
            title_text='Wert',
            row=1,
            col=1,
            side='right',
            showgrid=True,
            zeroline=False,
            automargin=True,
            ticklabelposition='outside right',
            # Erweiterte Interaktivität
            fixedrange=False,
            constrain='domain',
            scaleanchor='x'
        )

        fig.update_yaxes(
            title_text='Drawdown %',
            row=2,
            col=1,
            side='right',
            showgrid=True,
            zeroline=False,
            automargin=True,
            ticklabelposition='outside right',
            # Erweiterte Interaktivität
            fixedrange=False,
            constrain='domain'
        )

        # X-Achse konfigurieren
        fig.update_xaxes(
            range=[start_date, end_date],
            showspikes=True,  # Spike-Linien für Fadenkreuz
            spikemode='across',
            spikesnap='cursor',
            showline=True,
            showgrid=True,
            # Erweiterte Interaktivität
            fixedrange=False,
            constrain='domain'
        )

        return fig