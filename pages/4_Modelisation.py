import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fairness import demographic_parity_difference, disparate_impact_ratio

st.set_page_config(page_title="Modélisation", page_icon="🤖", layout="wide")


@st.cache_data
def load_and_prepare():
    df = pd.read_csv("healtcarecleaned.csv")
    df = df[df["gender"] != "Other"].copy()
    df = df.drop(columns=["id"])

    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = [
        "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
        "gender_enc", "ever_married_enc", "work_type_enc", "Residence_type_enc", "smoking_status_enc",
    ]
    X = df[feature_cols]
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    test_idx = X_test.index
    gender_test = df.loc[test_idx, "gender"].values
    residence_test = df.loc[test_idx, "Residence_type"].values

    return X_train, X_test, y_train, y_test, gender_test, residence_test, feature_cols


@st.cache_resource
def train_model(model_name, X_train, y_train):
    if model_name == "BalancedRandomForest (recommandé)":
        model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    elif model_name == "Random Forest + SMOTE":
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_res, y_res)
    else:
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_res, y_res)
    return model


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 Modélisation")
st.markdown("---")

X_train, X_test, y_train, y_test, gender_test, residence_test, feature_cols = load_and_prepare()

st.sidebar.header("Paramètres du modèle")
model_name = st.sidebar.selectbox(
    "Algorithme",
    ["BalancedRandomForest (recommandé)", "Random Forest + SMOTE", "Logistic Regression + SMOTE"],
)

model = train_model(model_name, X_train, y_train)
y_pred = model.predict(X_test)

# ── Explication stratégie anti-déséquilibre ───────────────────────────────────
with st.expander("Pourquoi corriger le déséquilibre de classes ?", expanded=False):
    st.markdown("""
    Le dataset est **très déséquilibré** : ~95% de patients sans AVC, ~5% avec AVC.
    Sans correction, le modèle prédit toujours "Pas d'AVC" → 95% d'accuracy mais **0 AVC détecté**.

    Trois stratégies sont disponibles :

    | Algorithme | Recall AVC | AVC détectés / 50 | F1 |
    |---|---|---|---|
    | Random Forest sans correction | 0.00 | 0 | 0.00 |
    | Random Forest + SMOTE | 0.18 | 9 | 0.15 |
    | **BalancedRandomForest** | **0.80** | **40** | **0.26** |

    **BalancedRandomForest** rééquilibre automatiquement les classes à chaque arbre, sans
    générer de données synthétiques. C'est la méthode la plus robuste pour ce dataset.
    """)

st.markdown("---")

# ── Performances globales ─────────────────────────────────────────────────────
st.header("1. Performances Globales")

acc   = accuracy_score(y_test, y_pred)
prec  = precision_score(y_test, y_pred, zero_division=0)
rec   = recall_score(y_test, y_pred, zero_division=0)
f1    = f1_score(y_test, y_pred, zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy",  f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall",    f"{rec:.3f}", help="Proportion d'AVC réels correctement détectés")
col4.metric("F1-Score",  f"{f1:.3f}")

# Matrice de confusion globale
st.subheader("Matrice de Confusion Globale")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

labels = ["Pas d'AVC", "AVC"]
fig_cm = px.imshow(
    cm, text_auto=True,
    x=labels, y=labels,
    color_continuous_scale="Blues",
    title="Matrice de confusion — toutes classes",
    labels={"x": "Prédit", "y": "Réel"},
)
st.plotly_chart(fig_cm, use_container_width=True)

col_tn, col_fp, col_fn, col_tp = st.columns(4)
col_tn.metric("Vrais Négatifs (TN)", tn,  help="Pas AVC prédit correctement")
col_fp.metric("Faux Positifs  (FP)", fp,  help="Pas AVC prédit comme AVC")
col_fn.metric("Faux Négatifs  (FN)", fn,  help="AVC manqué — dangereux !")
col_tp.metric("Vrais Positifs (TP)", tp,  help="AVC détecté correctement")

if fn > 0:
    st.error(f"**{fn} AVC non détectés (Faux Négatifs)** — ces patients n'auraient reçu aucune alerte.")
if tp > 0:
    st.success(f"**{tp} AVC correctement détectés (Vrais Positifs)** sur {fn+tp} cas réels.")

st.markdown("---")

# ── Importance des features ───────────────────────────────────────────────────
if model_name == "Random Forest":
    st.header("2. Importance des Variables")
    importances = pd.DataFrame({
        "Variable": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        importances, x="Importance", y="Variable", orientation="h",
        title="Importance des variables (Random Forest)",
        color="Importance", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("---")

# ── Métriques de fairness sur les prédictions ─────────────────────────────────
st.header("3. Métriques de Fairness sur les Prédictions")

res_dpd_gender = demographic_parity_difference(
    y_true=y_test.values, y_pred=y_pred, sensitive_attribute=gender_test
)
res_dir_gender = disparate_impact_ratio(
    y_true=y_test.values, y_pred=y_pred, sensitive_attribute=gender_test,
    unprivileged_value="Female", privileged_value="Male",
)
res_dpd_geo = demographic_parity_difference(
    y_true=y_test.values, y_pred=y_pred, sensitive_attribute=residence_test
)
res_dir_geo = disparate_impact_ratio(
    y_true=y_test.values, y_pred=y_pred, sensitive_attribute=residence_test,
    unprivileged_value="Rural", privileged_value="Urban",
)

col_f1, col_f2, col_f3, col_f4 = st.columns(4)
col_f1.metric("Parite Demo. Genre",    f"{res_dpd_gender['difference']:.4f}")
col_f2.metric("Ratio DI Genre (F/M)",  f"{res_dir_gender['ratio']:.4f}")
col_f3.metric("Parite Demo. Zone",     f"{res_dpd_geo['difference']:.4f}")
col_f4.metric("Ratio DI Zone (R/U)",   f"{res_dir_geo['ratio']:.4f}")

fairness_data = []
for g, r in res_dpd_gender["rates"].items():
    fairness_data.append({"Groupe": g, "Type": "Genre", "Taux predit (%)": r * 100})
for g, r in res_dpd_geo["rates"].items():
    fairness_data.append({"Groupe": g, "Type": "Zone", "Taux predit (%)": r * 100})

fig_fair = px.bar(
    pd.DataFrame(fairness_data),
    x="Groupe", y="Taux predit (%)", color="Type", barmode="group",
    title="Taux de prediction d'AVC positif par groupe sensible",
    text_auto=".2f",
)
st.plotly_chart(fig_fair, use_container_width=True)

st.markdown("---")

# ── Performances par groupe sensible ─────────────────────────────────────────
st.header("4. Performances par Groupe Sensible")


def group_metrics(y_true, y_pred, groups):
    rows = []
    for g in np.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "Groupe": g,
            "N": int(mask.sum()),
            "Accuracy":  round(accuracy_score(yt, yp), 3),
            "Precision": round(precision_score(yt, yp, zero_division=0), 3),
            "Recall":    round(recall_score(yt, yp, zero_division=0), 3),
            "F1":        round(f1_score(yt, yp, zero_division=0), 3),
        })
    return pd.DataFrame(rows)


tab1, tab2 = st.tabs(["Par Genre", "Par Zone de Résidence"])

with tab1:
    df_gp = group_metrics(y_test.values, y_pred, gender_test)
    st.dataframe(df_gp, use_container_width=True)
    fig_gp = px.bar(
        df_gp.melt(id_vars="Groupe", value_vars=["Accuracy", "Precision", "Recall", "F1"]),
        x="variable", y="value", color="Groupe", barmode="group",
        title="Metriques de performance par genre",
        labels={"variable": "Metrique", "value": "Score"},
    )
    st.plotly_chart(fig_gp, use_container_width=True)

with tab2:
    df_rp = group_metrics(y_test.values, y_pred, residence_test)
    st.dataframe(df_rp, use_container_width=True)
    fig_rp = px.bar(
        df_rp.melt(id_vars="Groupe", value_vars=["Accuracy", "Precision", "Recall", "F1"]),
        x="variable", y="value", color="Groupe", barmode="group",
        title="Metriques de performance par zone de residence",
        labels={"variable": "Metrique", "value": "Score"},
    )
    st.plotly_chart(fig_rp, use_container_width=True)

st.markdown("---")

# ── Confusion matrices par groupe ─────────────────────────────────────────────
st.header("5. Matrices de Confusion par Groupe")


def plot_confusion(y_true, y_pred, group_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Pas d'AVC", "AVC"]
    return px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale="Blues",
        title=f"Matrice de confusion — {group_name}",
        labels={"x": "Predit", "y": "Reel"},
    )


tab3, tab4 = st.tabs(["Par Genre", "Par Zone de Résidence"])

with tab3:
    cols = st.columns(len(np.unique(gender_test)))
    for i, g in enumerate(np.unique(gender_test)):
        mask = gender_test == g
        with cols[i]:
            st.plotly_chart(plot_confusion(y_test.values[mask], y_pred[mask], g), use_container_width=True)

with tab4:
    cols = st.columns(len(np.unique(residence_test)))
    for i, g in enumerate(np.unique(residence_test)):
        mask = residence_test == g
        with cols[i]:
            st.plotly_chart(plot_confusion(y_test.values[mask], y_pred[mask], g), use_container_width=True)
