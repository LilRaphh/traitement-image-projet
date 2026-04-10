def classify_deforestation(area_t0: dict, area_t1: dict, loss_percentage: float) -> dict:
    """
    Classe l'évolution observée entre t0 et t1 à partir des surfaces végétalisées.

    Les seuils restent volontairement simples et lisibles pour une restitution métier.
    """
    pct_t0 = float(area_t0["percentage"])
    pct_t1 = float(area_t1["percentage"])
    delta_points = round(pct_t1 - pct_t0, 2)
    delta_abs = abs(delta_points)

    if pct_t0 < 1.0 and pct_t1 < 1.0:
        label = "couverture_tres_faible"
        title = "Couverture végétale très faible"
        severity = "faible"
        summary = (
            "La scène contient très peu de végétation sur les deux dates, ce qui limite "
            "l'interprétation d'une dynamique forestière."
        )
    elif loss_percentage < -10:
        label = "reboisement_marque"
        title = "Reboisement marqué"
        severity = "positive"
        summary = "La couverture végétale progresse nettement entre t0 et t1."
    elif loss_percentage < -2:
        label = "reboisement_leger"
        title = "Reboisement léger"
        severity = "positive"
        summary = "La couverture végétale augmente légèrement entre t0 et t1."
    elif loss_percentage <= 2 and delta_abs <= 2:
        label = "stable"
        title = "Situation stable"
        severity = "faible"
        summary = "La couverture végétale reste globalement stable entre les deux dates."
    elif loss_percentage <= 10:
        label = "deforestation_legere"
        title = "Déforestation légère"
        severity = "moderee"
        summary = "On observe un recul limité mais réel de la végétation."
    elif loss_percentage <= 25:
        label = "deforestation_moderee"
        title = "Déforestation modérée"
        severity = "moderee"
        summary = "La perte de couverture végétale est nette et mérite une attention particulière."
    else:
        label = "deforestation_severe"
        title = "Déforestation sévère"
        severity = "elevee"
        summary = "La perte de couverture végétale est forte sur la zone étudiée."

    confidence = 0.55 + min(abs(loss_percentage) / 40.0, 0.25) + min(delta_abs / 20.0, 0.15)
    if pct_t0 < 5 or pct_t1 < 5:
        confidence -= 0.08
    confidence = round(max(0.5, min(confidence, 0.95)), 2)

    return {
        "label": label,
        "title": title,
        "severity": severity,
        "confidence": confidence,
        "summary": summary,
        "metrics": {
            "vegetation_t0_percentage": pct_t0,
            "vegetation_t1_percentage": pct_t1,
            "change_points": delta_points,
            "relative_loss_percentage": float(loss_percentage),
        },
    }
