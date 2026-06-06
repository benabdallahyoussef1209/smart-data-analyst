import streamlit as st

st.set_page_config(
    page_title="Smart Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CACHER LA NAVIGATION AUTOMATIQUE STREAMLIT ───────────────────────
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none !important; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Smart Analyst")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "📊 EDA Automatique", "🤖 ML Prediction", "💬 Chatbot Analyst"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    if "df" in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        st.success(f"✅ {st.session_state.get('filename', 'dataset.csv')}")
        st.caption(f"{df.shape[0]:,} lignes · {df.shape[1]} colonnes")
    else:
        st.info("Aucun fichier chargé")

# ── ROUTING ──────────────────────────────────────────────────────────
if page == "🏠 Accueil":
    from pages.home import show
    show()
elif page == "📊 EDA Automatique":
    from pages.eda import show
    show()
elif page == "🤖 ML Prediction":
    from pages.ml import show
    show()
elif page == "💬 Chatbot Analyst":
    from pages.chatbot import show
    show()