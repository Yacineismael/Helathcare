import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier

st.set_page_config(page_title="Prédiction Personnalisée", page_icon="🩺", layout="wide")


@st.cache_resource
def get_model():
    df = pd.read_csv("healtcarecleaned.csv")
    df = df[df["gender"] != "Other"].copy().drop(columns=["id"])

    encoders = {}
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = [
        "age", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
        "gender_enc", "ever_married_enc", "work_type_enc", "Residence_type_enc", "smoking_status_enc",
    ]
    X, y = df[feature_cols], df["stroke"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, encoders, feature_cols


model, encoders, feature_cols = get_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🩺 Prédiction du Risque d'AVC")
st.markdown("Renseignez les informations du patient ci-dessous pour obtenir une estimation du risque d'AVC.")
st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Informations démographiques")
    gender = st.selectbox("Genre", ["Male", "Female"])
    age = st.slider("Âge", min_value=1, max_value=82, value=45)
    ever_married = st.selectbox("Déjà marié(e) ?", ["Yes", "No"])
    work_type = st.selectbox(
        "Type d'emploi",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    )
    residence = st.selectbox("Zone de résidence", ["Urban", "Rural"])

with col_right:
    st.subheader("Informations médicales")
    hypertension = st.selectbox("Hypertension artérielle ?", ["Non", "Oui"])
    heart_disease = st.selectbox("Maladie cardiaque ?", ["Non", "Oui"])
    avg_glucose = st.number_input(
        "Glycémie moyenne (mg/dL)", min_value=50.0, max_value=300.0, value=90.0, step=0.5
    )
    bmi = st.number_input(
        "IMC (kg/m²)", min_value=10.0, max_value=100.0, value=25.0, step=0.1
    )
    smoking_status = st.selectbox(
        "Statut tabagique",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
    )

st.markdown("---")

# ── Prédiction ────────────────────────────────────────────────────────────────
if st.button("Lancer la prédiction", type="primary", use_container_width=True):

    # Encodage des valeurs saisies
    input_dict = {
        "age": age,
        "hypertension": 1 if hypertension == "Oui" else 0,
        "heart_disease": 1 if heart_disease == "Oui" else 0,
        "avg_glucose_level": avg_glucose,
        "bmi": bmi,
        "gender_enc":         encoders["gender"].transform([gender])[0],
        "ever_married_enc":   encoders["ever_married"].transform([ever_married])[0],
        "work_type_enc":      encoders["work_type"].transform([work_type])[0],
        "Residence_type_enc": encoders["Residence_type"].transform([residence])[0],
        "smoking_status_enc": encoders["smoking_status"].transform([smoking_status])[0],
    }

    X_input = pd.DataFrame([input_dict])[feature_cols]
    proba = model.predict_proba(X_input)[0][1]
    prediction = model.predict(X_input)[0]

    st.markdown("---")
    st.subheader("Résultat de la prédiction")

    col_res, col_gauge = st.columns([1, 2])

    with col_res:
        if prediction == 1:
            st.error("**Risque d'AVC détecté**")
        else:
            st.success("**Faible risque d'AVC**")

        st.metric("Probabilité estimée d'AVC", f"{proba * 100:.1f}%")

        if proba < 0.3:
            niveau = "Faible"
            couleur = "green"
        elif proba < 0.6:
            niveau = "Modéré"
            couleur = "orange"
        else:
            niveau = "Élevé"
            couleur = "red"

        st.markdown(f"**Niveau de risque :** :{couleur}[{niveau}]")

    with col_gauge:
        import plotly.graph_objects as go

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            number={"suffix": "%"},
            title={"text": "Probabilité d'AVC"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#E53935" if proba >= 0.5 else "#43A047"},
                "steps": [
                    {"range": [0, 30],  "color": "#C8E6C9"},
                    {"range": [30, 60], "color": "#FFF9C4"},
                    {"range": [60, 100],"color": "#FFCDD2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Récapitulatif du profil
    st.markdown("---")
    st.subheader("Récapitulatif du profil saisi")
    recap = {
        "Genre": gender, "Âge": age, "Marié(e)": ever_married,
        "Emploi": work_type, "Zone": residence,
        "Hypertension": hypertension, "Maladie cardiaque": heart_disease,
        "Glycémie (mg/dL)": avg_glucose, "IMC": bmi, "Tabac": smoking_status,
    }
    st.table(pd.DataFrame(recap.items(), columns=["Variable", "Valeur"]))

    st.caption(
        "Avertissement : Cette prédiction est fournie à titre indicatif uniquement. "
        "Elle ne remplace pas un diagnostic médical professionnel."
    )
