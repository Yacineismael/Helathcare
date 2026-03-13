import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Exploration des Données", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv("healtcarecleaned.csv")


df = load_data()

st.title("📊 Exploration des Données")
st.markdown("---")

# ── KPIs ──────────────────────────────────────────────────────────────────────
st.header("Métriques Clés")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Âge moyen", f"{df['age'].mean():.1f} ans")
col2.metric("IMC médian", f"{df['bmi'].median():.1f} kg/m²")
col3.metric("Glycémie moyenne", f"{df['avg_glucose_level'].mean():.1f} mg/dL")
col4.metric("Taux d'hypertension", f"{df['hypertension'].mean() * 100:.1f}%")

st.markdown("---")

# ── Filtres interactifs ───────────────────────────────────────────────────────
st.sidebar.header("Filtres")

selected_gender = st.sidebar.multiselect(
    "Genre", options=df["gender"].unique().tolist(), default=df["gender"].unique().tolist()
)
selected_age = st.sidebar.slider(
    "Tranche d'âge", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max()))
)
selected_residence = st.sidebar.multiselect(
    "Zone de résidence", options=df["Residence_type"].unique().tolist(), default=df["Residence_type"].unique().tolist()
)

df_f = df[
    df["gender"].isin(selected_gender)
    & df["age"].between(selected_age[0], selected_age[1])
    & df["Residence_type"].isin(selected_residence)
].copy()

df_f["stroke_label"] = df_f["stroke"].map({0: "Pas d'AVC", 1: "AVC"})

st.caption(f"Données filtrées : **{len(df_f):,}** patients affichés")
st.markdown("---")

# ── Visualisation 1 : Distribution variable cible ────────────────────────────
st.header("Visualisation 1 — Distribution de la Variable Cible")
fig1 = px.histogram(
    df_f, x="stroke_label", color="stroke_label",
    color_discrete_map={"Pas d'AVC": "#2196F3", "AVC": "#F44336"},
    title="Nombre de patients avec et sans AVC",
    labels={"stroke_label": "Statut AVC"},
    text_auto=True,
)
fig1.update_layout(showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

# ── Visualisation 2 : AVC par genre ──────────────────────────────────────────
st.header("Visualisation 2 — Taux d'AVC par Genre")
fig2 = px.bar(
    df_f.groupby(["gender", "stroke_label"]).size().reset_index(name="count"),
    x="gender", y="count", color="stroke_label", barmode="group",
    color_discrete_map={"Pas d'AVC": "#2196F3", "AVC": "#F44336"},
    title="Répartition des AVC par genre",
    labels={"gender": "Genre", "count": "Nombre de patients", "stroke_label": "Statut"},
)
st.plotly_chart(fig2, use_container_width=True)

# ── Visualisation 3 : Distribution âge par statut AVC ────────────────────────
st.header("Visualisation 3 — Distribution de l'Âge selon le Statut AVC")
fig3 = px.box(
    df_f, x="stroke_label", y="age", color="stroke_label",
    color_discrete_map={"Pas d'AVC": "#2196F3", "AVC": "#F44336"},
    title="Distribution de l'âge selon le statut AVC",
    labels={"stroke_label": "Statut AVC", "age": "Âge"},
    points="outliers",
)
fig3.update_layout(showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

# ── Visualisation 4 : Heatmap de corrélation ─────────────────────────────────
st.header("Visualisation 4 — Heatmap des Corrélations")
numeric_cols = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
fig4 = px.imshow(
    df_f[numeric_cols].corr().round(2),
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Matrice de corrélation des variables numériques",
    zmin=-1, zmax=1,
)
st.plotly_chart(fig4, use_container_width=True)

# ── Visualisation 5 : AVC par zone de résidence ───────────────────────────────
st.header("Visualisation 5 — Taux d'AVC par Zone de Résidence")
res_stroke = df_f.groupby("Residence_type")["stroke"].mean().reset_index()
res_stroke["Taux d'AVC (%)"] = res_stroke["stroke"] * 100
fig5 = px.bar(
    res_stroke, x="Residence_type", y="Taux d'AVC (%)", color="Residence_type",
    title="Taux d'AVC (%) selon la zone de résidence",
    labels={"Residence_type": "Zone de résidence"},
    text_auto=".2f",
)
fig5.update_layout(showlegend=False)
st.plotly_chart(fig5, use_container_width=True)

# ── Visualisation 6 : Scatter glycémie vs IMC ────────────────────────────────
st.header("Visualisation 6 — Glycémie vs IMC")
fig6 = px.scatter(
    df_f.sample(min(1000, len(df_f)), random_state=42),
    x="avg_glucose_level", y="bmi", color="stroke_label",
    color_discrete_map={"Pas d'AVC": "#2196F3", "AVC": "#F44336"},
    opacity=0.6,
    title="Relation entre glycémie et IMC (échantillon de 1000 patients)",
    labels={"avg_glucose_level": "Glycémie moyenne (mg/dL)", "bmi": "IMC (kg/m²)", "stroke_label": "Statut AVC"},
    trendline="ols",
)
st.plotly_chart(fig6, use_container_width=True)
