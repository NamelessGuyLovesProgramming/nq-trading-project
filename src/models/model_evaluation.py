def evaluate_model_quality(metrics):
    """
    Bewertet die Qualität eines Modells basierend auf seinen Metriken.

    Args:
        metrics (dict): Dictionary mit den Metriken (mse, rmse, mae, mape)

    Returns:
        tuple: (Qualitätsbewertung als String, Erklärung, Bewertungsscore von 0-10)
    """
    if not metrics or not all(k in metrics for k in ['mse', 'rmse', 'mae']):
        return "Unbekannt", "Nicht genügend Metriken vorhanden für eine Bewertung", 0

    # Bewertungskriterien (diese Werte müssen möglicherweise angepasst werden)
    # basierend auf typischen Werten für Finanzmarktdaten
    mse = metrics.get('mse', 0)
    rmse = metrics.get('rmse', 0)
    mae = metrics.get('mae', 0)
    mape = metrics.get('mape', 0) if metrics.get('mape') is not None else 100

    # Punktesystem (0-10)
    # Beachte: Niedrigere Werte sind besser für alle Fehlermetriken
    score = 0

    # MSE-Bewertung (0-2 Punkte)
    if mse < 0.00001:
        score += 2
    elif mse < 0.0001:
        score += 1.5
    elif mse < 0.001:
        score += 1
    elif mse < 0.01:
        score += 0.5

    # RMSE-Bewertung (0-3 Punkte)
    if rmse < 0.001:
        score += 3
    elif rmse < 0.005:
        score += 2.5
    elif rmse < 0.01:
        score += 2
    elif rmse < 0.05:
        score += 1
    elif rmse < 0.1:
        score += 0.5

    # MAE-Bewertung (0-3 Punkte)
    if mae < 0.0005:
        score += 3
    elif mae < 0.001:
        score += 2.5
    elif mae < 0.005:
        score += 2
    elif mae < 0.01:
        score += 1
    elif mae < 0.05:
        score += 0.5

    # MAPE-Bewertung (0-2 Punkte)
    if mape < 1:
        score += 2
    elif mape < 2:
        score += 1.5
    elif mape < 5:
        score += 1
    elif mape < 10:
        score += 0.5

    # Qualitätsbewertung basierend auf Gesamtpunktzahl
    if score >= 8:
        quality = "Ausgezeichnet"
        explanation = """
        Das Modell zeigt eine hervorragende Vorhersagegenauigkeit mit sehr niedrigen Fehlerraten.
        Es sollte in der Lage sein, Marktbewegungen mit hoher Präzision vorherzusagen.
        """
    elif score >= 6:
        quality = "Sehr gut"
        explanation = """
        Das Modell hat eine gute Vorhersagegenauigkeit mit niedrigen Fehlerraten.
        Es sollte zuverlässige Signale für Handelsentscheidungen liefern können.
        """
    elif score >= 4:
        quality = "Gut"
        explanation = """
        Das Modell zeigt eine vernünftige Vorhersagegenauigkeit.
        Es kann nützliche Handelssignale liefern, sollte aber mit weiteren Indikatoren kombiniert werden.
        """
    elif score >= 2:
        quality = "Mittelmäßig"
        explanation = """
        Das Modell hat eine mäßige Vorhersagegenauigkeit mit erhöhten Fehlerraten.
        Es kann als zusätzlicher Indikator dienen, aber nicht als alleinige Entscheidungsgrundlage.
        """
    else:
        quality = "Verbesserungswürdig"
        explanation = """
        Das Modell zeigt eine niedrige Vorhersagegenauigkeit mit hohen Fehlerraten.
        Es sollte überarbeitet werden, indem andere Features oder Hyperparameter verwendet werden.
        """

    return quality, explanation, score


def explain_metrics_in_plain_language(metrics):
    """
    Erklärt ML-Metriken in verständlicher Sprache.

    Args:
        metrics (dict): Dictionary mit den Metriken (mse, rmse, mae, mape)

    Returns:
        dict: Dictionary mit Metrik-Namen und Erklärungen
    """
    explanations = {}

    # MSE (Mean Squared Error)
    mse = metrics.get('mse', None)
    if mse is not None:
        if mse < 0.0001:
            mse_quality = "sehr niedrig (ausgezeichnet)"
        elif mse < 0.001:
            mse_quality = "niedrig (sehr gut)"
        elif mse < 0.01:
            mse_quality = "moderat (gut)"
        elif mse < 0.1:
            mse_quality = "erhöht (mittelmäßig)"
        else:
            mse_quality = "hoch (verbesserungswürdig)"

        explanations["MSE"] = f"""
        Der mittlere quadratische Fehler ist {mse_quality} bei {mse:.6f}.
        Dies ist ein Maß für die durchschnittliche quadrierte Abweichung zwischen den 
        vorhergesagten und tatsächlichen Werten. Kleiner ist besser!
        """

    # RMSE (Root Mean Squared Error)
    rmse = metrics.get('rmse', None)
    if rmse is not None:
        if rmse < 0.001:
            rmse_quality = "sehr niedrig (ausgezeichnet)"
        elif rmse < 0.01:
            rmse_quality = "niedrig (sehr gut)"
        elif rmse < 0.05:
            rmse_quality = "moderat (gut)"
        elif rmse < 0.1:
            rmse_quality = "erhöht (mittelmäßig)"
        else:
            rmse_quality = "hoch (verbesserungswürdig)"

        explanations["RMSE"] = f"""
        Die Wurzel des mittleren quadratischen Fehlers ist {rmse_quality} bei {rmse:.6f}.
        Dies ist ein Maß für die typische Größe des Vorhersagefehlers in denselben Einheiten 
        wie die Originaldaten. Kleiner ist besser!
        """

    # MAE (Mean Absolute Error)
    mae = metrics.get('mae', None)
    if mae is not None:
        if mae < 0.001:
            mae_quality = "sehr niedrig (ausgezeichnet)"
        elif mae < 0.005:
            mae_quality = "niedrig (sehr gut)"
        elif mae < 0.01:
            mae_quality = "moderat (gut)"
        elif mae < 0.05:
            mae_quality = "erhöht (mittelmäßig)"
        else:
            mae_quality = "hoch (verbesserungswürdig)"

        explanations["MAE"] = f"""
        Der mittlere absolute Fehler ist {mae_quality} bei {mae:.6f}.
        Dies ist ein Maß für die durchschnittliche absolute Abweichung zwischen den 
        vorhergesagten und tatsächlichen Werten. Kleiner ist besser!
        """

    # MAPE (Mean Absolute Percentage Error)
    mape = metrics.get('mape', None)
    if mape is not None:
        if mape < 1:
            mape_quality = "sehr niedrig (ausgezeichnet)"
        elif mape < 3:
            mape_quality = "niedrig (sehr gut)"
        elif mape < 7:
            mape_quality = "moderat (gut)"
        elif mape < 15:
            mape_quality = "erhöht (mittelmäßig)"
        else:
            mape_quality = "hoch (verbesserungswürdig)"

        explanations["MAPE"] = f"""
        Der mittlere absolute prozentuale Fehler ist {mape_quality} bei {mape:.2f}%.
        Dies gibt an, um wie viel Prozent die Vorhersagen im Durchschnitt von den 
        tatsächlichen Werten abweichen. Kleiner ist besser!
        """

    return explanations


def explain_model_architecture(model_info):
    """
    Erklärt die Modellarchitektur in verständlicher Sprache.

    Args:
        model_info (dict): Modell-Metadaten

    Returns:
        str: Verständliche Erklärung der Modellarchitektur
    """
    if 'parameters' not in model_info or 'layers' not in model_info.get('parameters', {}):
        return "Keine detaillierten Informationen zur Modellarchitektur verfügbar."

    layers = model_info.get('parameters', {}).get('layers', [])
    window_size = model_info.get('window_size', 0)

    # Zähle LSTM- und Dense-Schichten
    lstm_layers = [l for l in layers if 'lstm' in l.lower()]
    dense_layers = [l for l in layers if 'dense' in l.lower()]
    dropout_layers = [l for l in layers if 'dropout' in l.lower()]

    explanation = f"""
    ### Modellarchitektur - einfach erklärt 📊

    Dieses Modell wurde entwickelt, um Kursbewegungen basierend auf historischen Daten vorherzusagen:

    ℹ️ **Grundprinzip**: 
    Das Modell analysiert Sequenzen von {window_size} aufeinanderfolgenden Zeitpunkten, 
    um den nächsten Wert (meist den Schlusskurs) vorherzusagen.

    ℹ️ **Aufbau**:
    - Das Modell hat insgesamt {len(layers)} Schichten
    - {len(lstm_layers)} LSTM-Schicht(en): Diese "erinnern" sich an Muster in den Daten
    - {len(dense_layers)} Dense-Schicht(en): Diese treffen die eigentliche Vorhersage
    - {len(dropout_layers)} Dropout-Schicht(en): Diese verhindern Überanpassung

    ℹ️ **Komplexität**:
    Mit insgesamt {model_info.get('parameters', {}).get('total_params', 0):,} trainierbaren Parametern 
    ist dieses Modell {"komplex" if model_info.get('parameters', {}).get('total_params', 0) > 10000 else "moderat komplex" if model_info.get('parameters', {}).get('total_params', 0) > 1000 else "relativ einfach"}.

    ℹ️ **Wie es funktioniert**:
    Das Modell nimmt {window_size} Zeitpunkte mit jeweils {model_info.get('input_shape', [])[1] if len(model_info.get('input_shape', [])) > 1 else "?"} Features 
    (wie z.B. Open, Close, RSI, etc.) und versucht, ein Muster zu erkennen. 
    Es "lernt" aus den historischen Daten, wie sich diese Muster auf zukünftige Preisbewegungen auswirken.
    """

    return explanation

