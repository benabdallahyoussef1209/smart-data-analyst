import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os

# Charger la clé API depuis .env
load_dotenv()

def show():
    st.title("💬 Chatbot Data Analyst — IA")
    st.markdown("Posez vos questions en langage naturel — propulsé par **LLaMA 3.3 70B**")
    st.markdown("---")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Aucun dataset chargé. Retournez à l'Accueil.")
        return

    df = st.session_state.df

    # ── INITIALISER L'HISTORIQUE ─────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    f"👋 Bonjour ! Je suis votre Data Analyst IA propulsé par LLaMA 3.3.\n\n"
                    f"J'ai analysé **{st.session_state.get('filename', 'votre dataset')}** — "
                    f"**{df.shape[0]:,} lignes** et **{df.shape[1]} colonnes**.\n\n"
                    f"Posez-moi n'importe quelle question sur vos données !"
                )
            }
        ]

    # ── LAYOUT ───────────────────────────────────────────────────────
    col_chat, col_side = st.columns([3, 1])

    with col_side:
        st.markdown("#### 💡 Questions suggérées")
        suggestions = [
            "Décris-moi ce dataset",
            "Quelles colonnes ont des valeurs manquantes ?",
            "Quelles features utiliser pour le ML ?",
            "Y a-t-il des outliers ?",
            "Quelle est la distribution de age ?",
            "Quelles corrélations importantes ?",
            "Comment nettoyer ce dataset ?",
            "Quel modèle ML recommandes-tu ?",
        ]
        for q in suggestions:
            if st.button(q, key=f"sug_{q[:15]}", use_container_width=True):
                st.session_state.pending = q

        st.markdown("---")
        if st.button("🗑️ Effacer", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("#### 📋 Dataset")
        st.markdown(f"""
        **{st.session_state.get('filename', 'dataset')}**
        - 📏 {df.shape[0]:,} lignes
        - 📊 {df.shape[1]} colonnes
        - ❓ {df.isnull().sum().sum():,} manquantes
        """)

    with col_chat:
        # Afficher l'historique
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])

        # Traiter question suggérée
        if "pending" in st.session_state:
            question = st.session_state.pop("pending")
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("🤔 LLaMA analyse vos données..."):
                reponse = _appeler_groq(question, df)
            st.session_state.chat_history.append({"role": "assistant", "content": reponse})
            st.rerun()

        # Champ de saisie
        user_input = st.chat_input("Posez votre question ici...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("🤔 LLaMA analyse vos données..."):
                reponse = _appeler_groq(user_input, df)
            st.session_state.chat_history.append({"role": "assistant", "content": reponse})
            st.rerun()


# ── CONTEXTE DONNÉES POUR LE PROMPT ─────────────────────────────────
def _construire_contexte(df: pd.DataFrame) -> str:
    """Résume le dataset en texte pour le donner à LLaMA."""

    num_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(include=["object", "category"])

    # Statistiques descriptives
    stats = num_df.describe().round(3).to_string() if not num_df.empty else "Aucune"

    # Valeurs manquantes
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_str = missing.to_string() if not missing.empty else "Aucune"

    # Aperçu des premières lignes
    apercu = df.head(5).to_string()

    # Corrélations
    if num_df.shape[1] >= 2:
        corr = num_df.corr().round(3).to_string()
    else:
        corr = "Pas assez de colonnes numériques"

    contexte = f"""
Tu es un expert Data Analyst. Voici les informations complètes sur le dataset que l'utilisateur analyse :

=== INFORMATIONS GÉNÉRALES ===
- Nom du fichier : {st.session_state.get('filename', 'dataset')}
- Nombre de lignes : {df.shape[0]:,}
- Nombre de colonnes : {df.shape[1]}
- Colonnes numériques ({len(num_df.columns)}) : {', '.join(num_df.columns.tolist())}
- Colonnes catégorielles ({len(cat_df.columns)}) : {', '.join(cat_df.columns.tolist())}
- Doublons : {df.duplicated().sum()}

=== STATISTIQUES DESCRIPTIVES ===
{stats}

=== VALEURS MANQUANTES ===
{missing_str}

=== APERÇU DES DONNÉES (5 premières lignes) ===
{apercu}

=== MATRICE DE CORRÉLATION ===
{corr}

=== INSTRUCTIONS ===
- Réponds toujours en français
- Sois précis et utilise les vraies valeurs du dataset
- Utilise du markdown pour formater (gras, listes, code)
- Donne des recommandations concrètes et actionnables
- Si on te demande du code Python, fournis-le avec pandas
"""
    return contexte


# ── APPEL API GROQ ───────────────────────────────────────────────────
def _appeler_groq(question: str, df: pd.DataFrame) -> str:
    """Envoie la question + contexte à LLaMA via Groq API."""

    try:
        # Initialiser le client Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "❌ Clé API Groq manquante. Vérifiez votre fichier `.env`."

        client = Groq(api_key=api_key)

        # Construire le contexte complet du dataset
        contexte = _construire_contexte(df)

        # Appel API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Modèle LLaMA 3.3 70B
            messages=[
                {
                    "role": "system",
                    "content": contexte  # Contexte du dataset en system prompt
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.3,   # Bas = réponses plus précises/factuelles
            max_tokens=1024,   # Longueur max de la réponse
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Erreur API Groq : {str(e)}"