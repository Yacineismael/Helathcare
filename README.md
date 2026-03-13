# 🧠 Stroke Prediction — Détection de Biais

Application web interactive développée avec **Streamlit** pour analyser le dataset *Stroke Prediction* et détecter des biais algorithmiques liés au genre et à la zone géographique.

---

## Contexte

L'accident vasculaire cérébral (AVC) est l'une des principales causes de décès et de handicap dans le monde. Ce projet s'inscrit dans le **Parcours A — Détection de Biais** et utilise le dataset [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) (5 110 patients).

**Problématique :** Prédire le risque d'AVC tout en analysant les biais algorithmiques liés au **genre** et à la **zone géographique (Rural/Urban)**.

---

## Structure du projet

```
├── app.py                            # Page 1 — Accueil
├── pages/
│   ├── 2_Exploration_des_Donnees.py  # Page 2 — Exploration & visualisations
│   ├── 3_Detection_de_Biais.py       # Page 3 — Détection de biais (fairness)
│   ├── 4_Modelisation.py             # Page 4 — Modélisation & évaluation
│   └── 5_Prediction.py              # Page 5 — Prédiction personnalisée
├── utils/
│   └── fairness.py                   # Métriques de fairness (DPD, DI ratio)
├── healtcarecleaned.csv              # Dataset nettoyé (BMI imputé)
├── requirements.txt
└── README.md
```

---

## Pages de l'application

### 🏠 Page 1 — Accueil
- Présentation du projet et contexte médical
- Distribution de la variable cible (`stroke`)
- Aperçu interactif du dataset
- Description des colonnes

### 📊 Page 2 — Exploration des Données
- 4 KPIs clés (âge moyen, IMC médian, glycémie, taux d'hypertension)
- 2 filtres interactifs (genre, âge, zone de résidence)
- 6 visualisations : distribution de la cible, AVC par genre, distribution de l'âge, heatmap de corrélation, AVC par zone, scatter glycémie vs IMC

### ⚠️ Page 3 — Détection de Biais
- Analyse des biais de **genre** et de **zone géographique**
- **Métrique 1 :** Parité Démographique (différence de taux d'AVC entre groupes)
- **Métrique 2 :** Impact Disproportionné (ratio entre groupes non-privilégié / privilégié)
- Visualisations et interprétation des biais détectés

### 🤖 Page 4 — Modélisation
- Comparaison de 3 algorithmes : **BalancedRandomForest**, Random Forest + SMOTE, Logistic Regression + SMOTE
- Matrice de confusion avec TN / FP / FN / TP explicités
- Métriques de fairness appliquées aux prédictions
- Performances par groupe sensible (genre, zone)
- Matrices de confusion par groupe

### 🩺 Page 5 — Prédiction Personnalisée
- Formulaire de saisie des données d'un patient
- Prédiction en temps réel du risque d'AVC
- Jauge de probabilité interactive
- Niveau de risque (Faible / Modéré / Élevé)

---

## Gestion du déséquilibre de classes

Le dataset est très déséquilibré (~95% sans AVC, ~5% avec AVC). Sans correction, le modèle ne détecte aucun AVC.

| Algorithme | Recall AVC | AVC détectés / 50 | F1 |
|---|---|---|---|
| Random Forest sans correction | 0.00 | 0 / 50 | 0.00 |
| Random Forest + SMOTE | 0.18 | 9 / 50 | 0.15 |
| **BalancedRandomForest** | **0.80** | **40 / 50** | **0.26** |

**BalancedRandomForest** est retenu comme modèle principal car il rééquilibre automatiquement les classes à chaque arbre sans générer de données synthétiques.

---

## Installation locale

```bash
# Cloner le repository
git clone https://github.com/Yacineismael/Helathcare.git
cd Helathcare

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

---

## Dépendances

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
plotly>=5.18.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
```

---

## Dataset

- **Source :** [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Taille :** 5 110 patients, 12 variables
- **Variable cible :** `stroke` (0 = pas d'AVC, 1 = AVC)
- **Preprocessing :** valeurs manquantes de `bmi` imputées par médiane groupée (genre + âge)
