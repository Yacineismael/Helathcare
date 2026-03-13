import numpy as np


def demographic_parity_difference(y_true, y_pred, sensitive_attribute):
    """
    Calcule la différence de parité démographique entre groupes.
    Mesure la différence de taux de prédictions positives entre groupes.
    """
    groups = np.unique(sensitive_attribute)
    rates = {}
    for g in groups:
        mask = sensitive_attribute == g
        if np.sum(mask) > 0:
            rates[g] = float(np.mean(y_pred[mask]))

    values = list(rates.values())
    difference = max(values) - min(values) if len(values) >= 2 else 0.0

    return {
        "rates": rates,
        "difference": difference,
    }


def disparate_impact_ratio(y_true, y_pred, sensitive_attribute, unprivileged_value, privileged_value):
    """
    Calcule le ratio d'impact disproportionné.
    Un ratio < 0.8 indique un biais défavorable pour le groupe non-privilégié.
    """
    unpriv_mask = sensitive_attribute == unprivileged_value
    priv_mask = sensitive_attribute == privileged_value

    unpriv_rate = float(np.mean(y_pred[unpriv_mask])) if np.sum(unpriv_mask) > 0 else 0.0
    priv_rate = float(np.mean(y_pred[priv_mask])) if np.sum(priv_mask) > 0 else 0.0

    ratio = unpriv_rate / priv_rate if priv_rate > 0 else 0.0

    return {
        "unprivileged_rate": unpriv_rate,
        "privileged_rate": priv_rate,
        "ratio": ratio,
    }
