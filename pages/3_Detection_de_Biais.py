import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.fairness import demographic_parity_difference, disparate_impact_ratio

st.set_page_config(page_title="Détection de Biais", page_icon="⚠️", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("healtcarecleaned.csv")
    return df[df["gender"] != "Other"].copy()


df = load_data()

st.title("⚠️ Détection de Biais")
st.markdown("---")

# ── Explication du biais ──────────────────────────────────────────────────────
st.header("1. Biais Analysés")

tab1, tab2 = st.tabs(["Biais de Genre", "Biais Géographique (Rural/Urban)"])

with tab1:
    st.markdown("""
    ### Attribut sensible : Genre (Male / Female)

    **Pourquoi c'est problématique ?**

    Un biais de genre dans un modèle de prédiction d'AVC peut conduire à des inégalités
    graves dans la prise en charge médicale. Si le modèle prédit un risque d'AVC plus
    fréquemment chez les hommes que chez les femmes (ou inversement), certains patients
    pourraient être sous-diagnostiqués et ne pas recevoir les traitements préventifs nécessaires.

    Les biais de genre en santé sont documentés : les femmes ont historiquement été
    sous-représentées dans les études cliniques, ce qui peut biaiser les algorithmes entraînés
    sur ces données.
    """)

with tab2:
    st.markdown("""
    ### Attribut sensible : Zone de résidence (Rural / Urban)

    **Pourquoi c'est problématique ?**

    Un biais géographique peut refléter des inégalités d'accès aux soins. Si le modèle
    prédit différemment selon la zone de résidence, les patients ruraux — qui ont souvent
    moins accès aux soins spécialisés — pourraient être encore plus défavorisés.

    Ce biais peut aussi refléter des inégalités socio-économiques : les zones rurales
    ont généralement des revenus plus faibles, moins de services médicaux et des modes de vie
    différents qui influencent le risque d'AVC.
    """)

st.markdown("---")

# ── Métriques de fairness — Genre ─────────────────────────────────────────────
st.header("2. Métriques de Fairness — Genre")

result_dpd_gender = demographic_parity_difference(
    y_true=df["stroke"].values,
    y_pred=df["stroke"].values,
    sensitive_attribute=df["gender"].values,
)
result_dir_gender = disparate_impact_ratio(
    y_true=df["stroke"].values,
    y_pred=df["stroke"].values,
    sensitive_attribute=df["gender"].values,
    unprivileged_value="Female",
    privileged_value="Male",
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Métrique 1 : Parité Démographique (Genre)")
    st.metric("Différence de Parité", f"{result_dpd_gender['difference']:.4f}",
              help="Différence entre taux d'AVC max et min entre genres. Idéalement proche de 0.")
    for group, rate in result_dpd_gender["rates"].items():
        st.write(f"- **{group}** : {rate*100:.2f}%")
    if result_dpd_gender["difference"] < 0.05:
        st.success("Parité acceptable (différence < 5%)")
    else:
        st.warning("Biais détecté (différence ≥ 5%)")

with col2:
    st.subheader("Métrique 2 : Impact Disproportionné (Genre)")
    st.metric("Ratio DI (Female / Male)", f"{result_dir_gender['ratio']:.4f}",
              help="Ratio taux AVC Femme / Homme. Acceptable entre 0.8 et 1.25.")
    st.write(f"- **Female** : {result_dir_gender['unprivileged_rate']*100:.2f}%")
    st.write(f"- **Male** : {result_dir_gender['privileged_rate']*100:.2f}%")
    if 0.8 <= result_dir_gender["ratio"] <= 1.25:
        st.success("Ratio acceptable (entre 0.8 et 1.25)")
    else:
        st.warning("Impact disproportionné détecté (ratio hors [0.8, 1.25])")

# Graphique genre
gender_rates_df = pd.DataFrame([
    {"Genre": g, "Taux d'AVC (%)": r * 100}
    for g, r in result_dpd_gender["rates"].items()
])
fig_g = px.bar(gender_rates_df, x="Genre", y="Taux d'AVC (%)", color="Genre",
               color_discrete_map={"Male": "#1565C0", "Female": "#C62828"},
               title="Taux d'AVC par genre (%)", text_auto=".2f")
fig_g.add_hline(y=df["stroke"].mean() * 100, line_dash="dash", line_color="gray",
                annotation_text="Moyenne globale")
st.plotly_chart(fig_g, use_container_width=True)

st.markdown("---")

# ── Métriques de fairness — Zone géographique ─────────────────────────────────
st.header("3. Métriques de Fairness — Zone Géographique")

result_dpd_geo = demographic_parity_difference(
    y_true=df["stroke"].values,
    y_pred=df["stroke"].values,
    sensitive_attribute=df["Residence_type"].values,
)
result_dir_geo = disparate_impact_ratio(
    y_true=df["stroke"].values,
    y_pred=df["stroke"].values,
    sensitive_attribute=df["Residence_type"].values,
    unprivileged_value="Rural",
    privileged_value="Urban",
)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Métrique 1 : Parité Démographique (Zone)")
    st.metric("Différence de Parité", f"{result_dpd_geo['difference']:.4f}")
    for group, rate in result_dpd_geo["rates"].items():
        st.write(f"- **{group}** : {rate*100:.2f}%")
    if result_dpd_geo["difference"] < 0.05:
        st.success("Parité acceptable (différence < 5%)")
    else:
        st.warning("Biais détecté (différence ≥ 5%)")

with col4:
    st.subheader("Métrique 2 : Impact Disproportionné (Zone)")
    st.metric("Ratio DI (Rural / Urban)", f"{result_dir_geo['ratio']:.4f}")
    st.write(f"- **Rural** : {result_dir_geo['unprivileged_rate']*100:.2f}%")
    st.write(f"- **Urban** : {result_dir_geo['privileged_rate']*100:.2f}%")
    if 0.8 <= result_dir_geo["ratio"] <= 1.25:
        st.success("Ratio acceptable (entre 0.8 et 1.25)")
    else:
        st.warning("Impact disproportionné détecté (ratio hors [0.8, 1.25])")

# Graphique zone
geo_rates_df = pd.DataFrame([
    {"Zone": g, "Taux d'AVC (%)": r * 100}
    for g, r in result_dpd_geo["rates"].items()
])
fig_geo = px.bar(geo_rates_df, x="Zone", y="Taux d'AVC (%)", color="Zone",
                 color_discrete_map={"Urban": "#1B5E20", "Rural": "#E65100"},
                 title="Taux d'AVC par zone de résidence (%)", text_auto=".2f")
fig_geo.add_hline(y=df["stroke"].mean() * 100, line_dash="dash", line_color="gray",
                  annotation_text="Moyenne globale")
st.plotly_chart(fig_geo, use_container_width=True)

st.markdown("---")

# ── Interprétation ────────────────────────────────────────────────────────────
st.header("4. Interprétation des Biais")

male_rate = result_dpd_gender["rates"].get("Male", 0) * 100
female_rate = result_dpd_gender["rates"].get("Female", 0) * 100
defavorise = "femmes" if female_rate < male_rate else "hommes"

st.subheader("Biais de Genre")
st.markdown(f"""
L'analyse révèle une différence de **{result_dpd_gender['difference']*100:.2f} points de pourcentage**
entre les taux d'AVC masculin ({male_rate:.2f}%) et féminin ({female_rate:.2f}%).
Un modèle entraîné sur ces données risque de **sous-estimer le risque d'AVC chez les {defavorise}**,
car ce groupe présente moins de cas positifs dans le dataset.

Ce biais pourrait conduire à un sous-diagnostic, retardant la mise en place de traitements préventifs.
Pour le réduire, on peut utiliser des techniques de **rééchantillonnage** (SMOTE) ou appliquer des
**poids de classe** lors de l'entraînement du modèle.
""")

rural_rate = result_dir_geo["unprivileged_rate"] * 100
urban_rate = result_dir_geo["privileged_rate"] * 100

st.subheader("Biais Géographique")
st.markdown(f"""
La différence de taux d'AVC entre zones rurales ({rural_rate:.2f}%) et urbaines ({urban_rate:.2f}%)
est de **{result_dpd_geo['difference']*100:.2f} points**. Ce biais géographique reflète probablement
des inégalités d'accès aux soins préventifs.

L'impact réel est que le modèle pourrait moins bien performer sur les patients ruraux, précisément
ceux qui ont le plus besoin d'un outil de détection précoce. Une solution serait d'enrichir le dataset
avec des données géographiques supplémentaires et d'appliquer une **calibration par zone**.
""")
