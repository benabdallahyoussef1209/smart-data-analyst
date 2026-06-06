import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── Modèles disponibles ──────────────────────────────────────────────
MODELES_CLF = {
    "🌲 Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "⚡ Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "📈 Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

MODELES_REG = {
    "🌲 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "⚡ Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "📐 Ridge Regression": Ridge(),
    "📏 Linear Regression": LinearRegression(),
}


def show():
    st.title("🤖 ML Prediction")
    st.markdown("Entraînement automatique · Évaluation · Importance des features")
    st.markdown("---")

    # Vérifier qu'un dataset est chargé
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Aucun dataset chargé. Retournez à l'Accueil.")
        return

    df = st.session_state.df

    # ── CONFIGURATION ────────────────────────────────────────────────
    st.markdown("### ⚙️ Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Choisir la variable à prédire
        target = st.selectbox(
            "🎯 Variable cible (à prédire)",
            df.columns,
            index=_deviner_target(df)  # Devine automatiquement
        )
        # Détecter si c'est classification ou régression
        task = _detecter_tache(df, target)
        if task == "clf":
            st.success("Tâche : **Classification**")
        else:
            st.info("Tâche : **Régression**")

    with col2:
        # Choisir les features (variables d'entrée)
        toutes_features = [c for c in df.columns if c != target]
        features = st.multiselect(
            "📌 Features à utiliser",
            toutes_features,
            default=toutes_features[:min(10, len(toutes_features))]
        )

    with col3:
        # Paramètres d'entraînement
        test_size = st.slider("📊 % données de test", 10, 40, 20)
        if task == "clf":
            modele_nom = st.selectbox("🤖 Modèle", list(MODELES_CLF.keys()))
        else:
            modele_nom = st.selectbox("🤖 Modèle", list(MODELES_REG.keys()))

    if not features:
        st.warning("⚠️ Sélectionnez au moins une feature.")
        return

    # ── BOUTON ENTRAÎNER ─────────────────────────────────────────────
    if st.button("🚀 Lancer l'entraînement", use_container_width=False):
        _entrainer(df, target, features, task, modele_nom, test_size / 100)


# ── PRÉPARATION DES DONNÉES ──────────────────────────────────────────
def _preparer_donnees(df, target, features, task):
    try:
        X = df[features].copy()
        y = df[target].copy()

        # Encoder la cible si classification avec du texte
        if task == "clf" and y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            st.session_state.label_encoder = le
        else:
            y = pd.to_numeric(y, errors="coerce").fillna(y.median())

        # Encoder les colonnes texte en chiffres
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        feature_names = X.columns.tolist()

        # Remplacer les valeurs manquantes par la médiane
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        return X, y, feature_names

    except Exception as e:
        st.error(f"Erreur de préparation : {e}")
        return None, None, None


# ── ENTRAÎNEMENT + AFFICHAGE ─────────────────────────────────────────
def _entrainer(df, target, features, task, modele_nom, test_size):

    with st.spinner("🔄 Préparation des données..."):
        X, y, feature_names = _preparer_donnees(df, target, features, task)

    if X is None:
        return

    # Diviser en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if task == "clf" else None
    )

    with st.spinner("⚡ Entraînement en cours..."):
        # Choisir le bon modèle
        if task == "clf":
            modele = MODELES_CLF[modele_nom]
        else:
            modele = MODELES_REG[modele_nom]

        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)

    st.success("✅ Modèle entraîné avec succès !")
    st.markdown("---")

    # Afficher les résultats selon la tâche
    if task == "clf":
        _resultats_classification(modele, X_train, y_train, X_test, y_test, y_pred, feature_names)
    else:
        _resultats_regression(modele, X_train, y_train, X_test, y_test, y_pred, feature_names)


# ── RÉSULTATS CLASSIFICATION ─────────────────────────────────────────
def _resultats_classification(modele, X_train, y_train, X_test, y_test, y_pred, feature_names):

    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(modele, X_train, y_train, cv=5, scoring="accuracy")

    tab1, tab2, tab3 = st.tabs([
        "📊 Métriques",
        "🗺️ Matrice de confusion",
        "🔍 Importance des features"
    ])

    with tab1:
        # 4 métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{acc:.3f}")
        with col2:
            st.metric("CV Moyenne", f"{cv_scores.mean():.3f}")
        with col3:
            st.metric("CV Min", f"{cv_scores.min():.3f}")
        with col4:
            st.metric("CV Max", f"{cv_scores.max():.3f}")

        # Graphique cross-validation
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=cv_scores,
            marker_color="#6366f1"
        ))
        fig.add_hline(
            y=cv_scores.mean(),
            line_dash="dash",
            line_color="#f43f5e",
            annotation_text=f"Moyenne: {cv_scores.mean():.3f}"
        )
        fig.update_layout(
            title="Scores Cross-Validation (5 folds)",
            height=300,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Rapport détaillé
        st.markdown("#### Rapport de classification")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df.iloc[:-3], use_container_width=True)

    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(set(y_test))
        fig = px.imshow(
            cm,
            text_auto=True,
            x=[str(l) for l in labels],
            y=[str(l) for l in labels],
            color_continuous_scale="Purples",
            title="Matrice de confusion",
            labels=dict(x="Prédit", y="Réel")
        )
        fig.update_layout(height=450, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        _afficher_importance(modele, feature_names)


# ── RÉSULTATS RÉGRESSION ─────────────────────────────────────────────
def _resultats_regression(modele, X_train, y_train, X_test, y_test, y_pred, feature_names):

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    cv   = cross_val_score(modele, X_train, y_train, cv=5, scoring="r2")

    tab1, tab2, tab3 = st.tabs([
        "📊 Métriques",
        "🎯 Prédit vs Réel",
        "🔍 Importance des features"
    ])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
        with col4:
            st.metric("CV R² moyen", f"{cv.mean():.4f}")

        # Distribution des résidus
        residuals = y_test - y_pred
        fig = px.histogram(
            residuals, nbins=40,
            title="Distribution des résidus",
            color_discrete_sequence=["#6366f1"],
            template="plotly_white",
            labels={"value": "Résidu"}
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode="markers",
            marker=dict(color="#6366f1", opacity=0.6, size=6),
            name="Prédictions"
        ))
        # Ligne de prédiction parfaite
        mn = min(min(y_test), min(y_pred))
        mx = max(max(y_test), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="#f43f5e", dash="dash"),
            name="Parfait"
        ))
        fig.update_layout(
            title="Valeurs prédites vs réelles",
            xaxis_title="Réel",
            yaxis_title="Prédit",
            height=450,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        _afficher_importance(modele, feature_names)


# ── IMPORTANCE DES FEATURES ──────────────────────────────────────────
def _afficher_importance(modele, feature_names):
    st.markdown("#### 🔍 Importance des variables")

    if hasattr(modele, "feature_importances_"):
        # Random Forest / Gradient Boosting → feature_importances_
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": modele.feature_importances_
        }).sort_values("Importance", ascending=True).tail(15)

        fig = px.bar(
            fi, x="Importance", y="Feature",
            orientation="h",
            title="Feature Importance",
            color_discrete_sequence=["#6366f1"],
            template="plotly_white"
        )
        fig.update_layout(height=max(300, len(fi) * 30))
        st.plotly_chart(fig, use_container_width=True)

        top = fi.iloc[-1]["Feature"]
        st.info(f"🏆 Feature la plus importante : **`{top}`** ({fi.iloc[-1]['Importance']:.3f})")

    elif hasattr(modele, "coef_"):
        # Régression linéaire / logistique → coefficients
        coef = modele.coef_.flatten() if modele.coef_.ndim > 1 else modele.coef_
        fi = pd.DataFrame({
            "Feature": feature_names[:len(coef)],
            "Coefficient": coef
        }).sort_values("Coefficient", ascending=True)

        colors = ["#f43f5e" if c < 0 else "#6366f1" for c in fi["Coefficient"]]
        fig = go.Figure(go.Bar(
            x=fi["Coefficient"], y=fi["Feature"],
            orientation="h",
            marker_color=colors
        ))
        fig.update_layout(
            title="Coefficients du modèle",
            height=max(300, len(fi) * 30),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("🔵 Bleu = effet positif · 🔴 Rouge = effet négatif")


# ── UTILITAIRES ──────────────────────────────────────────────────────
def _detecter_tache(df, target):
    """Classification si texte ou peu de valeurs uniques, sinon Régression."""
    col = df[target]
    if col.dtype == "object" or col.nunique() <= 10:
        return "clf"
    return "reg"


def _deviner_target(df):
    """Devine la colonne cible la plus probable."""
    priorite = ["survived", "target", "label", "class",
                 "churn", "price", "salary", "species"]
    cols_lower = {c.lower(): i for i, c in enumerate(df.columns)}
    for p in priorite:
        if p in cols_lower:
            return cols_lower[p]
    return len(df.columns) - 1  # Par défaut : dernière colonne