# Modifications for src/custom_html_writer.py

def write_html_with_custom_interaction(fig, file_path):
    """
    Schreibt ein Plotly-Figure in eine HTML-Datei mit benutzerdefinierten Interaktionen.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Das Plotly-Figure-Objekt
    file_path : str
        Pfad zur Ausgabedatei
    """
    import plotly.io as pio

    # Verbessere die Proportionen des Charts
    fig.update_layout(
        # Besseres Seitenverhältnis für lesbarere Candlesticks
        width=1200,
        height=800,
        # Verbessere die Darstellung
        margin=dict(l=50, r=60, b=80, t=50)
    )

    # Verbessere die X-Achse mit lesbaren Zeitformaten
    fig.update_xaxes(
        # Bessere Zeitformatierung
        tickformat="%H:%M\n%d.%m",
        # Verbessere die Tick-Dichte
        nticks=15,
        # Ausrichtung der Beschriftungen
        tickangle=0,
        # Besser sichtbare Ticks
        tickfont=dict(size=11),
        # Verbesserte Gridlines
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(200, 200, 200, 0.3)'
    )

    # Verbessere Y-Achse
    fig.update_yaxes(
        # Bessere Skalierung mit angemessenem Padding
        autorange=True,
        # Sichtbarere Ticks
        tickfont=dict(size=11),
        # Verbesserte Gridlines
        showgrid=True,
        gridwidth=0.5,
        gridcolor='rgba(200, 200, 200, 0.3)'
    )

    # Standard-HTML generieren
    html_content = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=True,
        config={
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': [
                'zoomIn2d',
                'zoomOut2d',
                'resetScale2d'
            ]
        }
    )

    # Verbesserter Custom-JS-Code für bessere Candlestick-Darstellung
    custom_js = """
        <script>
        // Verbesserte Interaktionen für Plotly-Charts
        window.addEventListener('load', function() {
            var gd = document.querySelector('.plotly-graph-div');
            if (!gd) return;

            // Speichere die ursprünglichen Bereiche für Reset-Funktionalität
            var originalRanges = {};

            function saveOriginalRanges() {
                try {
                    if (gd._fullLayout && gd._fullLayout.xaxis && gd._fullLayout.xaxis.range) {
                        originalRanges.xaxis = [...gd._fullLayout.xaxis.range];
                    }
                    if (gd._fullLayout && gd._fullLayout.yaxis && gd._fullLayout.yaxis.range) {
                        originalRanges.yaxis = [...gd._fullLayout.yaxis.range];
                    }
                    console.log("Ursprüngliche Bereiche gespeichert:", originalRanges);
                } catch (e) {
                    console.error("Konnte ursprüngliche Bereiche nicht speichern:", e);
                }
            }

            // Nach kurzer Verzögerung ausführen
            setTimeout(saveOriginalRanges, 1000);

            // Verbessere die Darstellung der Candlesticks
            function enhanceCandlestickAppearance() {
                // Finde alle Candlestick-Paths
                var candlestickPaths = document.querySelectorAll('.candlestick path');

                candlestickPaths.forEach(function(path) {
                    // Verbessere die Linienbreite
                    path.style.strokeWidth = "2px";

                    // Erhöhe die Deckkraft für bessere Sichtbarkeit
                    path.style.opacity = "1";

                    // Verstärke die Füllung 
                    if (path.getAttribute('fill') === 'red') {
                        path.style.fill = '#FF4136';
                    } else if (path.getAttribute('fill') === 'green') {
                        path.style.fill = '#2ECC40';
                    }
                });

                // Verbessere auch die Candlestick-Linien (Dochte)
                var lines = document.querySelectorAll('.scatterlayer .trace path');
                lines.forEach(function(line) {
                    line.style.strokeWidth = "1.5px";
                });
            }

            // Führe die Verbesserung initial und nach jeder Aktualisierung des Charts aus
            enhanceCandlestickAppearance();
            gd.on('plotly_afterplot', enhanceCandlestickAppearance);

            // Verbessere die Darstellung der Achsen
            function improveAxisAppearance() {
                // X-Achsen-Beschriftungen verbessern
                var xTicks = document.querySelectorAll('.xtick text');
                xTicks.forEach(function(tick) {
                    tick.style.fontWeight = "normal";
                    tick.style.fontSize = "11px";
                });

                // Y-Achsen-Beschriftungen verbessern
                var yTicks = document.querySelectorAll('.ytick text');
                yTicks.forEach(function(tick) {
                    tick.style.fontWeight = "normal";
                    tick.style.fontSize = "11px";
                });
            }

            // Führe die Verbesserung initial und nach jeder Aktualisierung des Charts aus
            improveAxisAppearance();
            gd.on('plotly_afterplot', improveAxisAppearance);

            // Optimiere Y-Achsen-Darstellung
            function optimizeYAxisRange() {
                if (!gd._fullLayout || !gd._fullLayout.yaxis) return;

                // Versuche, aktuelle Preisdaten zu ermitteln
                var minY = Infinity;
                var maxY = -Infinity;

                // Gehe durch alle sichtbaren Traces
                gd._fullData.forEach(function(trace) {
                    if (trace.type === 'candlestick' && trace.visible !== 'legendonly') {
                        // Finde aktuelle sichtbare X-Range
                        var xRange = gd._fullLayout.xaxis.range;
                        var startX = new Date(xRange[0]).getTime();
                        var endX = new Date(xRange[1]).getTime();

                        // Gehe durch alle Datenpunkte im sichtbaren Bereich
                        for (var i = 0; i < trace.x.length; i++) {
                            var x = new Date(trace.x[i]).getTime();
                            if (x >= startX && x <= endX) {
                                minY = Math.min(minY, trace.low[i]);
                                maxY = Math.max(maxY, trace.high[i]);
                            }
                        }
                    }
                });

                // Wenn wir gültige Werte haben, passe die Y-Achse an
                if (minY !== Infinity && maxY !== -Infinity) {
                    // Füge etwas Padding hinzu (5%)
                    var padding = (maxY - minY) * 0.05;
                    Plotly.relayout(gd, {
                        'yaxis.range': [minY - padding, maxY + padding]
                    });
                }
            }

            // Nach Zoom oder Pan die Y-Achse optimieren
            gd.on('plotly_relayout', function(eventdata) {
                // Nur anpassen, wenn X-Achsenbereich geändert wurde
                if (eventdata['xaxis.range'] || eventdata['xaxis.range[0]'] || eventdata['xaxis.autorange']) {
                    setTimeout(optimizeYAxisRange, 10);
                }
            });

            // Erweiterte Mausrad-Funktion mit korrektem X-Achsen-Verhalten
            gd.addEventListener('wheel', function(e) {
                // Nur fortfahren, wenn wir über dem Hauptchart sind
                if (e.target.tagName === 'rect' || 
                    e.target.tagName === 'svg' || 
                    e.target.classList.contains('main-svg')) {

                    e.preventDefault();

                    // Stellen Sie sicher, dass wir Zugriff auf die Achsen haben
                    if (!gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullLayout.xaxis.range) {
                        console.error("Konnte nicht auf Achsenbereiche zugreifen");
                        return;
                    }

                    var xaxis = gd._fullLayout.xaxis;

                    // Aktuelle Bereiche
                    var xmin = xaxis.range[0];
                    var xmax = xaxis.range[1];

                    // Stellen Sie sicher, dass die Bereiche numerisch oder als Datum sind, nicht als String
                    if (typeof xmin === 'string') {
                        xmin = new Date(xmin).getTime();
                    }
                    if (typeof xmax === 'string') {
                        xmax = new Date(xmax).getTime();
                    }

                    var xrange = xmax - xmin;

                    // Bestimme Zoom-Richtung
                    var direction = e.deltaY < 0 ? -1 : 1;
                    var ctrlKey = e.ctrlKey || e.metaKey;

                    // Berechne Mausposition relativ zum Chart
                    var rect = gd.getBoundingClientRect();
                    var x = e.clientX - rect.left;
                    var width = rect.width;
                    var xpercent = x / width;

                    // Spezielle Zoomfaktoren
                    var factor = ctrlKey ? 0.15 : 0.05;

                    // Neue Bereiche berechnen
                    var newXMin, newXMax;

                    if (ctrlKey) {
                        // STRG+Mausrad: Beidseitiges Zoom vom Cursor aus
                        var xcenter = xmin + xrange * xpercent;

                        if (direction < 0) { // Herauszoomen
                            // Bereite den Chart beidseitig aus
                            newXMin = xcenter - (xcenter - xmin) * (1 + factor);
                            newXMax = xcenter + (xmax - xcenter) * (1 + factor);
                        } else { // Hineinzoomen
                            // Ziehe den Chart beidseitig zusammen
                            newXMin = xcenter - (xcenter - xmin) * (1 - factor);
                            newXMax = xcenter + (xmax - xcenter) * (1 - factor);
                        }
                    } else {
                        // Standard Mausrad ohne STRG: Nur die linke Seite strecken/stauchen
                        if (direction < 0) { // Mausrad nach oben - entquetschen (nach links)
                            // Bewege nur die linke Grenze, rechte bleibt fixiert
                            newXMin = xmin - xrange * factor;
                            newXMax = xmax; // Rechte Grenze bleibt unverändert
                        } else { // Mausrad nach unten - quetschen (nach rechts)
                            // Bewege nur die linke Grenze, rechte bleibt fixiert
                            newXMin = xmin + xrange * factor;
                            newXMax = xmax; // Rechte Grenze bleibt unverändert
                        }
                    }

                    // Achsenbereiche aktualisieren
                    Plotly.relayout(gd, {'xaxis.range': [newXMin, newXMax]});

                    // Nach dem Zoom die Y-Achse optimieren
                    setTimeout(optimizeYAxisRange, 10);
                }
            }, {passive: false});

            // Standardmäßig Candlesticks aktivieren und andere Traces ausblenden
            function setupDefaultVisibility() {
                if (!gd._fullData) return;

                // Finde den Candlestick-Trace
                var candlestickIndex = -1;
                for (var i = 0; i < gd._fullData.length; i++) {
                    if (gd._fullData[i].type === 'candlestick') {
                        candlestickIndex = i;
                        break;
                    }
                }

                if (candlestickIndex >= 0) {
                    // Erstelle ein Array mit den Sichtbarkeitseinstellungen
                    var visibility = [];
                    for (var i = 0; i < gd._fullData.length; i++) {
                        // Nur Candlesticks standardmäßig sichtbar, Rest als 'legendonly'
                        visibility.push(i === candlestickIndex ? true : 'legendonly');
                    }

                    // Update die Sichtbarkeit
                    Plotly.restyle(gd, {visible: visibility});
                }

                // Optimiere auch die Y-Achse
                optimizeYAxisRange();
            }

            // Nach kurzer Verzögerung die Sichtbarkeit setzen
            setTimeout(setupDefaultVisibility, 500);

            // Füge Reset-Button hinzu
            var resetBtn = document.createElement('button');
            resetBtn.innerHTML = 'Reset Zoom';
            resetBtn.className = 'reset-zoom-btn';
            resetBtn.style.position = 'absolute';
            resetBtn.style.top = '10px';
            resetBtn.style.left = '10px';
            resetBtn.style.zIndex = '999';
            resetBtn.style.padding = '5px 10px';
            resetBtn.style.cursor = 'pointer';
            resetBtn.style.backgroundColor = '#f8f9fa';
            resetBtn.style.border = '1px solid #ddd';
            resetBtn.style.borderRadius = '4px';

            resetBtn.addEventListener('click', function() {
                if (originalRanges.xaxis) {
                    Plotly.relayout(gd, {
                        'xaxis.range': originalRanges.xaxis,
                        'yaxis.autorange': true  // Auto-Range für Y-Achse beim Reset
                    });
                    // Nach Reset auch Y-Achse optimieren
                    setTimeout(optimizeYAxisRange, 100);
                }
            });

            var container = gd.parentElement;
            if (container && !document.querySelector('.reset-zoom-btn')) {
                container.style.position = 'relative';
                container.appendChild(resetBtn);
            }
        });
        </script>

        <style>
        /* Verbesserte Darstellung der Candlesticks */
        .candlestick path {
            stroke-width: 2px !important;
            opacity: 1 !important;
        }

        /* Dickere Candlestick-Körper */
        .candlestick path[fill="green"], 
        .candlestick path[fill="red"] {
            stroke-width: 2px !important;
            fill-opacity: 1 !important;
        }

        /* Grün und Rot besser hervorheben */
        .candlestick path[fill="green"] {
            fill: #2ECC40 !important; /* Kräftigeres Grün für bessere Sichtbarkeit */
        }

        .candlestick path[fill="red"] {
            fill: #FF4136 !important; /* Leuchtendes Rot für bessere Sichtbarkeit */
        }

        /* Verbessere die Darstellung der Achsen */
        .xaxis path, .yaxis path {
            stroke-width: 1.5px !important;
            stroke: #555 !important;
        }

        /* Verbessere X-Achsen-Beschriftungen */
        .xtick text {
            font-size: 11px !important;
            font-weight: normal !important;
        }

        /* Verbessere Y-Achsen-Beschriftungen */
        .ytick text {
            font-size: 11px !important;
            font-weight: normal !important;
        }

        /* Optimierter Stil für Raster */
        .gridlayer path {
            stroke-width: 0.5px !important;
            stroke-opacity: 0.3 !important;
            stroke: #aaa !important;
        }

        /* Verbesserte Chart-Hintergrundfarbe */
        .main-svg .plotbg {
            fill: #FCFCFC !important;
        }

        /* Stil für Reset-Button */
        .reset-zoom-btn {
            font-family: Arial, sans-serif;
            font-size: 12px;
            font-weight: bold;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        }

        .reset-zoom-btn:hover {
            background-color: #e2e6ea;
        }
        </style>
        """

    # Füge das benutzerdefinierte JavaScript vor dem schließenden body-Tag ein
    html_content = html_content.replace('</body>', custom_js + '</body>')

    # Schreibe die modifizierte HTML-Datei
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Chart mit benutzerdefinierten Interaktionen gespeichert unter: {file_path}")