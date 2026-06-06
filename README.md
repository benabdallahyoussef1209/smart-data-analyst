# 🧠 Smart Data Analyst

> Application web intelligente d'analyse de données, propulsée par **LLaMA 3.3 70B** via Groq API.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3-green)

---

## 🎯 Fonctionnalités
application en ligne : [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-data-analyst-ldkdbsmegfgxheqbvykyox.streamlit.app)


| Page | Fonctionnalité |
|---|---|
| 🏠 **Accueil** | Upload CSV + aperçu automatique |
| 📊 **EDA Automatique** | Statistiques, distributions, corrélations, qualité |
| 🤖 **ML Prediction** | Entraînement, évaluation, feature importance |
| 💬 **Chatbot IA** | Questions en langage naturel sur vos données |

---

## 🛠️ Stack Technique

- **Frontend** : Streamlit
- **Data** : Pandas, NumPy
- **Visualisation** : Plotly
- **Machine Learning** : Scikit-learn
- **IA Générative** : LLaMA 3.3 70B via Groq API

---

## 🚀 Installation locale

### 1. Cloner le repo
```bash
git clone https://github.com/benabdallahyoussef1209/smart-data-analyst.git
cd smart-data-analyst
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API
Créez un fichier `.env` à la racine :
GROQ_API_KEY=votre_cle_groq_ici
Obtenez votre clé gratuite sur [console.groq.com](https://console.groq.com)

### 5. Lancer l'application
```bash
streamlit run app.py
```

---

## 📊 Datasets supportés

- ✅ CSV avec séparateur `,` ou `;`
- ✅ Datasets d'exemple intégrés : Titanic, Iris, Auto MPG, Diamonds
- ✅ Détection automatique du type de tâche (Classification / Régression)

---

## 🤖 Modèles ML disponibles

**Classification**
- Random Forest
- Gradient Boosting
- Logistic Regression

**Régression**
- Random Forest
- Gradient Boosting
- Ridge Regression
- Linear Regression

---

## 💬 Exemples de questions Chatbot
"Décris-moi ce dataset"
"Quelles colonnes ont des valeurs manquantes ?"
"Quel modèle ML recommandes-tu ?"
"Quelles corrélations importantes ?"
"Comment nettoyer ce dataset ?"

---

## 📁 Structure du projet
smart-data-analyst/
├── app.py              # Point d'entrée + navigation
├── requirements.txt    # Dépendances Python
├── .gitignore          # Fichiers exclus de Git
├── .env                # Clé API (non commité)
└── pages/
├── home.py         # Upload CSV + aperçu
├── eda.py          # Analyse exploratoire
├── ml.py           # Machine Learning
└── chatbot.py      # Chatbot IA (LLaMA 3.3)

---

## 👨‍💻 Auteur

**Youssef Ben Abdallah**  
[![GitHub](https://img.shields.io/badge/GitHub-benabdallahyoussef1209-black)](https://github.com/benabdallahyoussef1209)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Youssef-blue)](www.linkedin.com/in/youssef-ben-abdallah1209)

---

## 📄 Licence

MIT License — libre d'utilisation et de modification.
