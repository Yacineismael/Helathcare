import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
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

    # Encodage des variables catégorielles
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

    # Garder les colonnes sensibles pour le test
    test_idx = X_test.index
    gender_test = df.loc[test_idx, "gender"].values
    residence_test = df.loc[test_idx, "Residence_type"].values

    return X_train, X_test, y_train, y_test, gender_test, residence_test, feature_cols


@st.cache_resource
def train_model(model_name, X_train, y_train):
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    else:
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🤖 Modélisation")
st.markdown("---")

X_train, X_test, y_train, y_test, gender_test, residence_test, feature_cols = load_and_prepare()

st.sidebar.header("Paramètres du modèle")
model_name = st.sidebar.selectbox("Algorithme", ["Random Forest", "Logistic Regression"])

model = train_model(model_name, X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

# ── Performances globales ─────────────────────────────────────────────────────
st.header("1. Performances Globales")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("F1-Score", f"{f1:.3f}")

st.markdown("---")

# ── Importance des features (RF uniquement) ───────────────────────────────────
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

# Genre
res_dpd_gender = demographic_parity_difference(
    y_true=y_test.values, y_pred=y_pred, sensitive_attribute=gender_test
)
res_dir_gender = disparate_impact_ratio(
    y_true=y_test.values, y_pred=y_pred, sensitive_attribute=gender_test,
    unprivileged_value="Female", privileged_value="Male",
)
# Zone
res_dpd_geo = demographic_parity_difference(
    y_true=y_test.values, y_pred=y_pred, sensitive_attribute=residence_test
)
res_dir_geo = disparate_impact_ratio(
    y_true=y_test.values, y_pred=y_pred, sensitive_attribute=residence_test,
    unprivileged_value="Rural", privileged_value="Urban",
)

col_f1, col_f2, col_f3, col_f4 = st.columns(4)
col_f1.metric("Parité Démographique (Genre)", f"{res_dpd_gender['difference']:.4f}")
col_f2.metric("Ratio DI Genre (F/M)", f"{res_dir_gender['ratio']:.4f}")
col_f3.metric("Parité Démographique (Zone)", f"{res_dpd_geo['difference']:.4f}")
col_f4.metric("Ratio DI Zone (Rural/Urban)", f"{res_dir_geo['ratio']:.4f}")

# Graphique comparatif prédictions par groupe
st.subheader("Taux de prédiction d'AVC par groupe")

fairness_data = []
for g, r in res_dpd_gender["rates"].items():
    fairness_data.append({"Groupe": g, "Type": "Genre", "Taux prédit (%)" : r * 100})
for g, r in res_dpd_geo["rates"].items():
    fairness_data.append({"Groupe": g, "Type": "Zone", "Taux prédit (%)": r * 100})

fig_fair = px.bar(
    pd.DataFrame(fairness_data),
    x="Groupe", y="Taux prédit (%)", color="Type", barmode="group",
    title="Taux de prédiction d'AVC positif par groupe sensible",
    text_auto=".2f",
)
st.plotly_chart(fig_fair, use_container_width=True)

st.markdown("---")

# ── Performances par groupe sensible ─────────────────────────────────────────
st.header("4. Performances par Groupe Sensible")

tab1, tab2 = st.tabs(["Par Genre", "Par Zone de Résidence"])

def group_metrics(y_true, y_pred, groups):
    rows = []
    for g in np.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        rows.append({
            "Groupe": g,
            "N": int(mask.sum()),
            "Accuracy": round(accuracy_score(yt, yp), 3),
            "Precision": round(precision_score(yt, yp, zero_division=0), 3),
            "Recall": round(recall_score(yt, yp, zero_division=0), 3),
            "F1": round(f1_score(yt, yp, zero_division=0), 3),
        })
    return pd.DataFrame(rows)

with tab1:
    df_gender_perf = group_metrics(y_test.values, y_pred, gender_test)
    st.dataframe(df_gender_perf, use_container_width=True)

    fig_g_perf = px.bar(
        df_gender_perf.melt(id_vars="Groupe", value_vars=["Accuracy", "Precision", "Recall", "F1"]),
        x="variable", y="value", color="Groupe", barmode="group",
        title="Métriques de performance par genre",
        labels={"variable": "Métrique", "value": "Score"},
    )
    st.plotly_chart(fig_g_perf, use_container_width=True)

with tab2:
    df_geo_perf = group_metrics(y_test.values, y_pred, residence_test)
    st.dataframe(df_geo_perf, use_container_width=True)

    fig_geo_perf = px.bar(
        df_geo_perf.melt(id_vars="Groupe", value_vars=["Accuracy", "Precision", "Recall", "F1"]),
        x="variable", y="value", color="Groupe", barmode="group",
        title="Métriques de performance par zone de résidence",
        labels={"variable": "Métrique", "value": "Score"},
    )
    st.plotly_chart(fig_geo_perf, use_container_width=True)

st.markdown("---")

# ── Confusion matrices par groupe ─────────────────────────────────────────────
st.header("5. Matrices de Confusion par Groupe")

def plot_confusion(y_true, y_pred, group_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Pas d'AVC", "AVC"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        title=f"Matrice de confusion — {group_name}",
        labels={"x": "Prédit", "y": "Réel"},
    )
    return fig

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
