import streamlit as st
import pandas as pd
import io

def show():
    st.title("🏠 Smart Data Analyst")
    st.markdown("Uploadez votre CSV pour commencer l'analyse.")
    st.markdown("---")

    # ── SECTION UPLOAD ──────────────────────────────────────
    uploaded = st.file_uploader(
        "📂 Glissez votre fichier CSV ici",
        type=["csv"],
        help="Séparateur virgule ou point-virgule détecté automatiquement"
    )

    if uploaded is not None:
        try:
            # Lire le contenu brut
            content = uploaded.read()

            # Détecter le séparateur automatiquement
            sample = content[:2000].decode("utf-8", errors="replace")
            sep = ";" if sample.count(";") > sample.count(",") else ","

            # Créer le DataFrame
            df = pd.read_csv(io.BytesIO(content), sep=sep)

            # Stocker dans session_state (accessible par toutes les pages)
            st.session_state.df = df
            st.session_state.filename = uploaded.name

            st.success(f"✅ **{uploaded.name}** chargé avec succès !")

            # Afficher l'aperçu
            _afficher_apercu(df)

        except Exception as e:
            st.error(f"❌ Erreur de lecture : {e}")

    # ── SECTION DATASETS D'EXEMPLE ──────────────────────────
    st.markdown("---")
    st.markdown("### Ou testez avec un dataset d'exemple")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🏠 Titanic (Classification)", use_container_width=True):
            _charger_exemple(
                "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
                "titanic.csv"
            )

    with col2:
        if st.button("🌸 Iris (Multi-class)", use_container_width=True):
            _charger_exemple(
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
                "iris.csv"
            )

    col3, col4 = st.columns(2)

    with col3:
        if st.button("🚗 Auto MPG (Régression)", use_container_width=True):
            _charger_exemple(
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv",
                "mpg.csv"
            )

    with col4:
        if st.button("💎 Diamonds (Régression)", use_container_width=True):
            _charger_exemple(
                "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
                "diamonds.csv"
            )

    # ── APERCU si dataset déjà chargé ───────────────────────
    if "df" in st.session_state and st.session_state.df is not None and uploaded is None:
        _afficher_apercu(st.session_state.df)


def _charger_exemple(url: str, nom: str):
    """Charge un dataset depuis une URL et le stocke."""
    with st.spinner(f"Chargement de {nom}..."):
        try:
            df = pd.read_csv(url)
            st.session_state.df = df
            st.session_state.filename = nom
            st.success(f"✅ {nom} chargé !")
            st.rerun()  # Recharge la page pour afficher l'aperçu
        except Exception as e:
            st.error(f"Erreur : {e}")


def _afficher_apercu(df: pd.DataFrame):
    """Affiche les métriques et le tableau d'aperçu."""
    st.markdown("---")
    st.markdown("### 👁 Aperçu du dataset")

    # ── 4 métriques clés ────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📏 Lignes", f"{df.shape[0]:,}")

    with col2:
        st.metric("📊 Colonnes", f"{df.shape[1]}")

    with col3:
        n_num = len(df.select_dtypes(include="number").columns)
        st.metric("🔢 Numériques", n_num)

    with col4:
        pct_missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        st.metric("❓ Valeurs manquantes", f"{pct_missing:.1f}%")

    # ── Tableau des premières lignes ────────────────────────
    st.markdown("#### Premières lignes")
    st.dataframe(df.head(10), use_container_width=True)

    # ── Types des colonnes ──────────────────────────────────
    st.markdown("#### Types des colonnes")
    types_df = pd.DataFrame({
        "Colonne": df.columns,
        "Type": df.dtypes.values.astype(str),
        "Valeurs uniques": df.nunique().values,
        "Manquantes": df.isnull().sum().values,
    })
    st.dataframe(types_df, use_container_width=True)

    # ── Message de guidance ─────────────────────────────────
    st.info("✅ Dataset prêt ! Naviguez vers **EDA Automatique** dans la sidebar pour analyser vos données.")