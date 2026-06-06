import streamlit as st      # Framework pour créer l'interface web
import pandas as pd         # Manipulation des données (DataFrame)
import numpy as np          # Calculs mathématiques (corrélations, etc.)
import plotly.express as px # Graphiques simples et rapides
import plotly.graph_objects as go  # Graphiques avancés (heatmap)

def show():
    # Titre et description de la page
    st.title("📊 EDA Automatique")
    st.markdown("Analyse exploratoire complète générée automatiquement.")
    st.markdown("---")  # Ligne de séparation visuelle

    # ── GARDE-FOU ────────────────────────────────────────────────────
    # Vérifie si un dataset a été chargé dans home.py
    # session_state = mémoire partagée entre toutes les pages
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Aucun dataset chargé. Retournez à l'Accueil.")
        return  # Arrête l'exécution ici si pas de données

    # Récupérer le DataFrame stocké depuis home.py
    df = st.session_state.df

    # Séparer les colonnes numériques (int, float)
    num_df = df.select_dtypes(include="number")

    # Séparer les colonnes texte/catégorielles
    cat_df = df.select_dtypes(include=["object", "category"])

    # ── ONGLETS ──────────────────────────────────────────────────────
    # Crée 5 onglets cliquables en haut de la page
    # Chaque variable (tab1...tab5) contrôle un onglet
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Vue d'ensemble",
        "📈 Distributions",
        "🔗 Corrélations",
        "🧹 Qualité",
        "💡 Insights"
    ])

    # Le contenu de chaque onglet est dans son bloc "with"
    with tab1:
        _vue_ensemble(df, num_df, cat_df)  # Appelle la fonction dédiée

    with tab2:
        _distributions(df, num_df, cat_df)

    with tab3:
        _correlations(num_df)

    with tab4:
        _qualite(df)

    with tab5:
        _insights(df, num_df, cat_df)


# ── FONCTION 1 : Vue d'ensemble ──────────────────────────────────────
def _vue_ensemble(df, num_df, cat_df):
    # 4 métriques côte à côte (comme des cartes KPI)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📏 Lignes", f"{df.shape[0]:,}")   # shape[0] = nombre de lignes
    with col2:
        st.metric("📊 Colonnes", f"{df.shape[1]}")   # shape[1] = nombre de colonnes
    with col3:
        st.metric("🔢 Numériques", len(num_df.columns))
    with col4:
        st.metric("🔤 Catégorielles", len(cat_df.columns))

    st.markdown("---")

    # Deux colonnes côte à côte
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Statistiques descriptives")
        if not num_df.empty:
            # describe() = count, mean, std, min, 25%, 50%, 75%, max
            # .T = transposer (colonnes deviennent lignes, plus lisible)
            # .round(3) = arrondir à 3 décimales
            st.dataframe(num_df.describe().T.round(3), use_container_width=True)

    with col_right:
        st.markdown("#### Répartition des types")

        # Compter combien de colonnes de chaque type (int64, float64, object...)
        type_counts = df.dtypes.astype(str).value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]

        # Graphique camembert (pie chart)
        fig = px.pie(
            type_counts, values="Count", names="Type",
            color_discrete_sequence=["#6366f1", "#8b5cf6", "#a78bfa"],
            template="plotly_white"
        )
        fig.update_layout(height=300, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Données brutes")
    # Affiche tout le DataFrame scrollable, limité à 250px de hauteur
    st.dataframe(df, use_container_width=True, height=250)


# ── FONCTION 2 : Distributions ───────────────────────────────────────
def _distributions(df, num_df, cat_df):
    # Layout : panneau de contrôle à gauche, graphique à droite
    col_left, col_right = st.columns([1, 2])  # [1,2] = ratio largeur

    with col_left:
        # L'utilisateur choisit le type de variable à visualiser
        col_type = st.radio("Type", ["Numérique", "Catégorielle"])

        if col_type == "Numérique" and not num_df.empty:
            # Liste déroulante avec toutes les colonnes numériques
            selected = st.selectbox("Colonne", num_df.columns)
            # Choix du type de graphique
            plot_type = st.radio("Graphique", ["Histogramme", "Box Plot", "Violin"])

        elif col_type == "Catégorielle" and not cat_df.empty:
            selected = st.selectbox("Colonne", cat_df.columns)
            plot_type = "Barres"  # Fixe pour les catégories
        else:
            st.info("Pas de colonnes disponibles.")
            return  # Arrête si aucune colonne du type choisi

    with col_right:
        if col_type == "Numérique":

            if plot_type == "Histogramme":
                fig = px.histogram(
                    df, x=selected,
                    nbins=40,          # Nombre de barres
                    marginal="box",    # Ajoute un box plot au-dessus
                    title=f"Distribution de {selected}",
                    color_discrete_sequence=["#6366f1"],
                    template="plotly_white"
                )

            elif plot_type == "Box Plot":
                # Box plot : montre médiane, quartiles, outliers
                fig = px.box(
                    df, y=selected,
                    title=f"Box Plot — {selected}",
                    color_discrete_sequence=["#6366f1"],
                    template="plotly_white"
                )

            else:  # Violin
                # Violin = Box Plot + densité de distribution
                fig = px.violin(
                    df, y=selected,
                    box=True,  # Inclut le box plot à l'intérieur
                    title=f"Violin — {selected}",
                    color_discrete_sequence=["#6366f1"],
                    template="plotly_white"
                )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Statistiques textuelles sous le graphique
            col = df[selected].dropna()  # Enlève les NaN pour les calculs
            st.info(
                f"Moyenne: **{col.mean():.2f}** | "
                f"Médiane: **{col.median():.2f}** | "
                f"Std: **{col.std():.2f}** | "
                f"Skewness: **{col.skew():.2f}**"
                # Skewness > 0 = asymétrie à droite, < 0 = à gauche
            )

        else:
            # Pour les catégories : compter les occurrences de chaque valeur
            vc = df[selected].value_counts().head(15)  # Top 15 modalités

            fig = px.bar(
                x=vc.index,   # Les catégories
                y=vc.values,  # Leur fréquence
                title=f"Modalités — {selected}",
                labels={"x": selected, "y": "Fréquence"},
                color_discrete_sequence=["#6366f1"],
                template="plotly_white"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# ── FONCTION 3 : Corrélations ────────────────────────────────────────
def _correlations(num_df):
    # Minimum 2 colonnes numériques pour calculer des corrélations
    if num_df.shape[1] < 2:
        st.info("Il faut au moins 2 colonnes numériques.")
        return

    # Matrice de corrélation de Pearson
    # Valeurs entre -1 (corrélation négative) et +1 (positive)
    # 0 = pas de corrélation
    corr = num_df.corr()
    corr_values = corr.values.copy() 

    # Heatmap = carte de chaleur colorée
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,              # Les valeurs numériques
        x=corr.columns.tolist(),    # Labels axe X
        y=corr.columns.tolist(),    # Labels axe Y
        colorscale="RdBu",          # Rouge=négatif, Bleu=positif
        zmid=0,                     # Centre la couleur sur 0
        text=corr.round(2).values,  # Texte affiché dans chaque cellule
        texttemplate="%{text}",     # Format du texte
        textfont_size=11,
    ))
    fig.update_layout(
        title="Matrice de corrélation",
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tableau des paires les plus corrélées
    st.markdown("#### Top corrélations")

    # unstack() = transformer la matrice en liste de paires
    corr_pairs = corr.abs().unstack()

    # Enlever les corrélations parfaites (variable avec elle-même = 1.0)
    corr_pairs = corr_pairs[corr_pairs < 1.0].drop_duplicates().sort_values(ascending=False)

    top = corr_pairs.head(8).reset_index()
    top.columns = ["Variable 1", "Variable 2", "Corrélation"]
    st.dataframe(top.round(3), use_container_width=True)


# ── FONCTION 4 : Qualité des données ────────────────────────────────
def _qualite(df):
    st.markdown("#### Analyse qualité")

    # Tableau récapitulatif de la qualité colonne par colonne
    quality = pd.DataFrame({
        "Colonne": df.columns,
        "Type": df.dtypes.values.astype(str),
        "Manquantes": df.isnull().sum().values,          # Nombre de NaN
        "% manquant": (df.isnull().mean() * 100).round(2).values,  # Pourcentage
        "Valeurs uniques": df.nunique().values,           # Cardinalité
    })
    st.dataframe(quality, use_container_width=True)

    # Graphique uniquement pour les colonnes qui ont des valeurs manquantes
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if not missing.empty:
        fig = px.bar(
            y=missing.index,   # Noms des colonnes
            x=missing.values,  # Nombre de valeurs manquantes
            orientation="h",   # Barres horizontales
            title="Valeurs manquantes par colonne",
            color_discrete_sequence=["#f43f5e"],  # Rouge pour alerter
            template="plotly_white"
        )
        # Hauteur dynamique selon le nombre de colonnes concernées
        fig.update_layout(height=max(200, len(missing) * 45))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ Aucune valeur manquante !")

    # Détecter les lignes identiques
    dupes = df.duplicated().sum()
    if dupes > 0:
        st.warning(f"⚠️ {dupes:,} lignes dupliquées ({dupes/len(df)*100:.1f}%)")
    else:
        st.success("✅ Aucun doublon !")


# ── FONCTION 5 : Insights automatiques ──────────────────────────────
def _insights(df, num_df, cat_df):
    st.markdown("#### 💡 Insights générés automatiquement")

    # Liste vide — on va la remplir avec les observations détectées
    insights = []

    # Insight 1 : taille du dataset
    insights.append(
        f"📦 Dataset : **{df.shape[0]:,} lignes** × **{df.shape[1]} colonnes**"
    )

    # Insight 2 : valeurs manquantes
    total_miss = df.isnull().sum().sum()  # Total sur tout le DataFrame
    if total_miss == 0:
        insights.append("✅ **Aucune valeur manquante** — dataset complet !")
    else:
        # Trouver la colonne la plus touchée
        worst = df.isnull().sum().idxmax()
        pct = df[worst].isnull().mean() * 100
        insights.append(
            f"⚠️ **{total_miss:,} valeurs manquantes**. "
            f"Colonne la plus affectée : `{worst}` ({pct:.1f}%)"
        )

    # Insight 3 : asymétrie des distributions (skewness)
    if not num_df.empty:
        skewed = num_df.skew().abs()  # Valeur absolue du skewness
        if (skewed > 2).any():        # Seuil : |skewness| > 2 = très asymétrique
            cols = skewed[skewed > 2].index.tolist()
            insights.append(
                f"📐 Distributions très asymétriques : "
                f"`{'`, `'.join(cols[:3])}` → transformation log recommandée"
            )

        # Insight 4 : forte corrélation (risque multicolinéarité)
        corr = num_df.corr().abs()
        corr_arr = corr.values.copy()   
        np.fill_diagonal(corr_arr,0)#mettre la diagonale à 0 (auto-corrélation)
        max_corr = corr_arr.max()     # Corrélation maximale trouvée
        if max_corr > 0.8:                 # Seuil : > 0.8 = forte corrélation
            idx = corr.stack().idxmax()    # Trouver quelle paire
            insights.append(
                f"🔗 Forte corrélation ({max_corr:.2f}) "
                f"entre `{idx[0]}` et `{idx[1]}`"
            )

    # Insight 5 : doublons
    dupes = df.duplicated().sum()
    if dupes > 0:
        insights.append(f"🔁 **{dupes:,} doublons** détectés — nettoyage recommandé")

    # Afficher chaque insight dans une boîte bleue
    for insight in insights:
        st.info(insight)

    st.markdown("---")
    st.success("🚀 Prochaine étape : **ML Prediction** !")