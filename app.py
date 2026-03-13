import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Stroke Prediction - Détection de Biais",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def load_data():
    return pd.read_csv("healtcarecleaned.csv")


df = load_data()

# ── Page 1 : Accueil ──────────────────────────────────────────────────────────

st.title("🧠 Stroke Prediction — Détection de Biais")
st.markdown("---")

st.header("Contexte")
st.markdown(
    """
L'accident vasculaire cérébral (AVC) est l'une des principales causes de décès et de handicap dans le monde.
Identifier les facteurs de risque de manière précoce permet d'améliorer la prévention et la prise en charge des patients.

Ce dataset contient des informations médicales et démographiques sur **5 110 patients**, collectées
pour prédire le risque d'AVC. La variable cible `stroke` indique si le patient a subi un AVC (1) ou non (0).

Notre analyse va au-delà de la simple prédiction : nous cherchons à détecter des **biais algorithmiques**
liés au **genre** et à la **zone géographique (Rural/Urban)**, qui pourraient conduire à des inégalités
dans le diagnostic ou le traitement médical.
"""
)

st.markdown("---")

st.header("Distribution de la Variable Cible")
stroke_counts = df["stroke"].value_counts().reset_index()
stroke_counts.columns = ["stroke", "count"]
stroke_counts["label"] = stroke_counts["stroke"].map({0: "Pas d'AVC", 1: "AVC"})

fig = px.pie(
    stroke_counts,
    names="label",
    values="count",
    color="label",
    color_discrete_map={"Pas d'AVC": "#2196F3", "AVC": "#F44336"},
    title="Répartition des cas d'AVC dans le dataset",
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "**Attention** : Le dataset est très déséquilibré (~95% sans AVC vs ~5% avec AVC). "
    "Cela doit être pris en compte lors de la modélisation et de l'évaluation des biais."
)

st.markdown("---")

st.header("Aperçu des Données")
st.dataframe(df.head(50), use_container_width=True)

st.header("Description des Colonnes")
col_desc = {
    "id": "Identifiant unique du patient",
    "gender": "Genre du patient (Male / Female / Other)",
    "age": "Âge du patient (en années)",
    "hypertension": "Hypertension artérielle (0 = Non, 1 = Oui)",
    "heart_disease": "Maladie cardiaque (0 = Non, 1 = Oui)",
    "ever_married": "Statut marital (Yes / No)",
    "work_type": "Type d'emploi (Private / Self-employed / Govt_job / children / Never_worked)",
    "Residence_type": "Zone de résidence (Urban / Rural)",
    "avg_glucose_level": "Taux moyen de glucose dans le sang (mg/dL)",
    "bmi": "Indice de masse corporelle (kg/m²)",
    "smoking_status": "Statut tabagique (formerly smoked / never smoked / smokes / Unknown)",
    "stroke": "Variable cible : AVC survenu (0 = Non, 1 = Oui)",
}
st.table(pd.DataFrame(list(col_desc.items()), columns=["Colonne", "Description"]))
