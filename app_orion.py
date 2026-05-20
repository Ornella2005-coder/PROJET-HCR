"""
PROJET O.R.I.O.N — Application Streamlit (vraies données)
HCR / UNHCR — Bureau Europe
Auteure : FANGNON Ornella — Bachelor 2 Data & BI — NEXA Digital School

Lancement :
    python -m streamlit run app_orion.py

Dépendances :
    pip install streamlit scikit-learn pandas numpy matplotlib seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from models_orion import (
    load_real_data,
    plot_correlation_matrix,
    plot_regplots,
    train_linear,
    train_logistic,
    train_classification,
    build_summary,
    FEATURES_NUM, TARGET_REG, TARGET_BIN, TARGET_CAT,
    HCR_NAVY, HCR_BLUE, HCR_SKY, HCR_GOLD, HCR_TEAL, HCR_RED,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="O.R.I.O.N — ML",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A2357 0%, #1A4A9B 100%);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,.88) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #F5A800 !important; font-weight: 600;
    font-size: .82rem; text-transform: uppercase; letter-spacing: .05em;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #F5A800 !important; }

.main .block-container { background: #F4F6FB; padding-top: 1.4rem; }

.hero {
    background: linear-gradient(135deg, #0A2357 0%, #1A4A9B 60%, #2E82D4 100%);
    border-radius: 14px; padding: 26px 34px; margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 18px;
}
.hero-logo {
    background: #F5A800; border-radius: 9px; padding: 11px 15px;
    font-weight: 800; font-size: 13px; color: #0A2357;
    line-height: 1.2; text-align: center; flex-shrink: 0;
}
.hero-title { font-size: 22px; font-weight: 700; color: #fff; margin: 0; }
.hero-sub { font-size: 12.5px; color: rgba(255,255,255,.6); margin: 4px 0 0; }
.hbadge {
    display: inline-block; background: rgba(255,255,255,.14);
    color: rgba(255,255,255,.85); font-size: 11px;
    padding: 3px 10px; border-radius: 20px;
    border: 1px solid rgba(255,255,255,.2); margin: 2px 3px 0 0;
}
.stab-title {
    font-size: 19px; font-weight: 700; color: #0A2357;
    border-left: 4px solid #F5A800; padding-left: 11px; margin-bottom: 4px;
}
.stab-desc { font-size: 12px; color: #4A5E82; margin-bottom: 14px; padding-left: 15px; }

div[data-testid="metric-container"] {
    background: white; border: 1px solid rgba(10,35,87,.10);
    border-radius: 10px; padding: 13px 15px !important;
    box-shadow: 0 1px 4px rgba(10,35,87,.07);
}
div[data-testid="metric-container"] label {
    color: #7B8FB5 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: .06em; font-weight: 600 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0A2357 !important; font-weight: 700 !important;
}

button[data-baseweb="tab"] { font-weight: 600 !important; font-size: 13px !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0A2357 !important; border-bottom: 3px solid #F5A800 !important;
}

.info-box {
    background: #EAF3FC; border-left: 4px solid #2E82D4;
    border-radius: 8px; padding: 11px 15px; font-size: 13px;
    color: #1A4A9B; margin-bottom: 11px;
}
.ok-box {
    background: #E0F5EF; border-left: 4px solid #00A878;
    border-radius: 8px; padding: 11px 15px; font-size: 13px;
    color: #085041; margin-bottom: 11px;
}
.warn-box {
    background: #FFF8E1; border-left: 4px solid #F5A800;
    border-radius: 8px; padding: 11px 15px; font-size: 13px;
    color: #7A5000; margin-bottom: 11px;
}
.err-box {
    background: #FDECEA; border-left: 4px solid #D94035;
    border-radius: 8px; padding: 11px 15px; font-size: 13px;
    color: #7A1A1A; margin-bottom: 11px;
}
hr { border-color: rgba(10,35,87,.10) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📂 Données")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Charger dataset_final.csv",
        type=["csv"],
        help="Fichier généré par analyse_hcr.ipynb dans output/dataset_final.csv"
    )

    st.markdown("---")
    st.markdown("## ⚙️ Paramètres")
    test_size = st.slider("Part de test (%)", 10, 30, 20, 5,
                          help="80/20 recommandé") / 100
    random_state = st.number_input("Random state", 0, 999, 42)

    st.markdown("---")
    st.markdown("## ✅ Modèles actifs")
    show_lin = st.checkbox("Régression Linéaire",     value=True)
    show_log = st.checkbox("Régression Logistique",   value=True)
    show_clf = st.checkbox("Classification (DT+KNN)", value=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:rgba(255,255,255,.6);line-height:1.8">
    <b style="color:#F5A800">Projet O.R.I.O.N</b><br>
    FANGNON Ornella<br>
    Bachelor 2 Data & BI<br>
    NEXA Digital School<br><br>
    <b style="color:#F5A800">Sources :</b><br>
    UNHCR · Eurostat · EM-DAT<br>
    Période : 2015–2023
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <div class="hero-logo">UN<br>HCR</div>
  <div>
    <div class="hero-title">Projet O.R.I.O.N — Apprentissage Supervisé</div>
    <div class="hero-sub">
      Vraies données · UNHCR Refugee Data Finder · Eurostat · EM-DAT ·
      Bureau Europe · 2015–2023
    </div>
    <div style="margin-top:8px">
      <span class="hbadge">Régression Linéaire → RMSE + R²</span>
      <span class="hbadge">Régression Logistique → Accuracy/F1</span>
      <span class="hbadge">Arbre de Décision</span>
      <span class="hbadge">KNN + Pondération + Normalisation</span>
      <span class="hbadge">Split 80/20</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_data(file_obj):
    return load_real_data(file_obj)

@st.cache_data(show_spinner=False)
def get_models(csv_bytes, ts, rs):
    """
    Cache basé sur le contenu du CSV + paramètres.
    csv_bytes : bytes du fichier pour que le cache se régénère si fichier change.
    """
    import io
    df = load_real_data(io.BytesIO(csv_bytes))
    lin = train_linear(df)
    log = train_logistic(df)
    clf = train_classification(df)
    return df, lin, log, clf


# ── Chargement ────────────────────────────────────────────────────────────────

if uploaded is None:
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Aucun fichier chargé.</b><br>
    Dans la sidebar → <b>"Charger dataset_final.csv"</b><br>
    Ce fichier est généré automatiquement par <code>analyse_hcr.ipynb</code>
    dans le dossier <code>output/</code>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Structure attendue du CSV :")
    st.code(
        "annee, pays_accueil, iso3_accueil, iso2_accueil, pays_origine, iso3_origine,\n"
        "dec_pos, dec_rej, dec_total, taux,\n"
        "nb_demandes, pib_hab, nb_cat, deces, affect",
        language="text"
    )
    st.info("👆 Lancez d'abord **analyse_hcr.ipynb** puis revenez ici avec le CSV généré.")
    st.stop()

# ── Entraînement ─────────────────────────────────────────────────────────────
with st.spinner("🔄 Chargement et entraînement des modèles..."):
    try:
        csv_bytes = uploaded.getvalue()
        df, lin, log, clf = get_models(csv_bytes, test_size, random_state)
    except Exception as e:
        st.markdown(f'<div class="err-box">❌ Erreur : {e}</div>',
                    unsafe_allow_html=True)
        st.stop()

(lin_pipe, lin_m, lin_df, lin_fig, lin_yte, lin_ypred) = lin
(log_pipe, log_m, log_df, log_fig, log_yte, log_ypred) = log
(clf_r, clf_df, clf_fig_cmp, clf_fig_cm, clf_yte)      = clf

st.markdown(
    f'<div class="ok-box">✅ <b>dataset_final.csv</b> chargé — '
    f'<b>{df.shape[0]:,}</b> lignes × <b>{df.shape[1]}</b> colonnes — '
    f'Taux moyen : <b>{df[TARGET_REG].mean():.1f} %</b> — '
    f'Split : <b>80 % entraînement / 20 % test</b></div>',
    unsafe_allow_html=True
)


# ═══════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ═══════════════════════════════════════════════════════════════════════════════

tab_data, tab_corr, tab_lin, tab_log, tab_clf, tab_recap = st.tabs([
    "📊 Données",
    "🔗 Corrélations & Regplots",
    "📈 Régression Linéaire",
    "⚖️ Régression Logistique",
    "🌳 Classification",
    "📋 Récapitulatif",
])


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 1 — DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

with tab_data:
    st.markdown('<div class="stab-title">Aperçu du dataset réel</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="stab-desc">dataset_final.csv — généré par analyse_hcr.ipynb '
        'à partir de UNHCR Refugee Data Finder · Eurostat · EM-DAT · 2015–2023</div>',
        unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Lignes",       f"{df.shape[0]:,}")
    c2.metric("Colonnes",     f"{df.shape[1]}")
    c3.metric("Taux moyen",   f"{df[TARGET_REG].mean():.1f} %")
    c4.metric("Taux min",     f"{df[TARGET_REG].min():.1f} %")
    c5.metric("Taux max",     f"{df[TARGET_REG].max():.1f} %")
    c6.metric("Années",       f"{int(df['annee'].min())}–{int(df['annee'].max())}")

    st.markdown("#### 🗂️ Extrait du dataset (50 premières lignes)")
    st.dataframe(df.head(50), use_container_width=True, height=320)

    st.markdown("#### 📐 Statistiques descriptives")
    st.dataframe(df[FEATURES_NUM + [TARGET_REG]].describe().round(2),
                 use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribution du taux d'acceptation")
        fig_d, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[TARGET_REG], bins=40, color=HCR_BLUE, alpha=0.8,
                edgecolor="white", linewidth=0.4)
        ax.axvline(df[TARGET_REG].mean(), color=HCR_GOLD, linewidth=2,
                   linestyle="--",
                   label=f"Moyenne : {df[TARGET_REG].mean():.1f} %")
        ax.axvline(df[TARGET_REG].median(), color=HCR_RED, linewidth=1.5,
                   linestyle=":",
                   label=f"Médiane : {df[TARGET_REG].median():.1f} %")
        ax.set_xlabel("Taux d'acceptation (%)", fontsize=10)
        ax.set_ylabel("Effectif", fontsize=10)
        ax.set_title("Distribution du taux d'acceptation (dataset réel)",
                      fontsize=10, color=HCR_NAVY, fontweight="bold")
        ax.legend(fontsize=9)
        fig_d.tight_layout()
        st.pyplot(fig_d)
        plt.close()

    with col2:
        st.markdown("#### Répartition des 3 classes (variable cible ML)")
        fig_c, ax2 = plt.subplots(figsize=(6, 3.5))
        counts  = df[TARGET_CAT].value_counts().sort_index()
        colors  = [HCR_TEAL, HCR_SKY, HCR_RED]
        bars    = ax2.bar(counts.index, counts.values,
                          color=colors[:len(counts)],
                          edgecolor="white", linewidth=0.4)
        for bar, cnt in zip(bars, counts.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, cnt + 5,
                     f"{cnt:,}", ha="center", fontsize=10,
                     fontweight="bold", color=HCR_NAVY)
        ax2.set_ylabel("Effectif", fontsize=10)
        ax2.set_title("Catégories : Faible (<15%) · Modéré (15–40%) · Élevé (>40%)",
                       fontsize=10, color=HCR_NAVY, fontweight="bold")
        fig_c.tight_layout()
        st.pyplot(fig_c)
        plt.close()

    st.markdown("#### Pays d'accueil dans le dataset")
    pays_stats = (
        df.groupby("pays_accueil")
        .apply(lambda x: pd.Series({
            "Nb lignes":      len(x),
            "Taux moyen (%)": round(x["dec_pos"].sum() / x["dec_total"].sum() * 100, 1),
            "PIB moyen (€)":  round(x["pib_hab"].mean(), 0),
        }))
        .sort_values("Taux moyen (%)", ascending=False)
        .reset_index()
    )
    st.dataframe(pays_stats, use_container_width=True, height=320)


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 2 — CORRÉLATIONS
# ─────────────────────────────────────────────────────────────────────────────

with tab_corr:
    st.markdown(
        '<div class="stab-title">Corrélations & Regplots</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="stab-desc">'
        'Analyse des relations entre les variables du projet ORION '
        'et le taux d\'acceptation.'
        '</div>',
        unsafe_allow_html=True
    )

    # ═══════════════════════════════════════════════════════════════════════
    # TABLEAU DES CORRÉLATIONS
    # ═══════════════════════════════════════════════════════════════════════

    corr_s = (
        df[FEATURES_NUM + [TARGET_REG]]
        .corr()[TARGET_REG]
        .drop(TARGET_REG)
    )

    corr_df = (
        corr_s
        .sort_values(key=abs, ascending=False)
        .reset_index()
    )

    corr_df.columns = ["Feature", "Corrélation"]

    corr_df["Force"] = corr_df["Corrélation"].abs().apply(
        lambda x:
            "🔴 Forte (>0.5)" if x > 0.5 else
            ("🟡 Modérée (0.2–0.5)" if x > 0.2 else
             "⚪ Faible (<0.2)")
    )

    corr_df["Hypothèse ORION"] = corr_df["Feature"].map({
        "pib_hab":     "PIB élevé → taux plus élevé",
        "annee":       "Effet temporel",
        "nb_demandes": "Volume des demandes",
        "dec_total":   "Volume des décisions",
        "nb_cat":      "Catastrophes naturelles",
        "deces":       "Décès liés aux catastrophes",
        "affect":      "Population affectée",
    })

    st.markdown("#### Corrélations avec le taux d'acceptation")

    st.dataframe(
        corr_df.style.background_gradient(
            subset=["Corrélation"],
            cmap="RdYlGn",
            vmin=-1,
            vmax=1
        ).format({
            "Corrélation": "{:.4f}"
        }),
        use_container_width=True,
        height=280
    )

    # ═══════════════════════════════════════════════════════════════════════
    # MATRICE DE CORRÉLATION
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("#### Matrice de corrélation complète")

    st.markdown(
        '<div class="info-box">'
        'La matrice permet de visualiser les relations positives ou négatives '
        'entre toutes les variables numériques du dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Génération de la matrice..."):
        fig_corr = plot_correlation_matrix(df)
        st.pyplot(fig_corr)
        plt.close()

    # ═══════════════════════════════════════════════════════════════════════
    # REGPLOTS
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown("#### Regplots — Variables vs Taux d'acceptation")

    st.markdown(
        '<div class="info-box">'
        'Chaque graphique montre la relation entre une variable et le '
        'taux d\'acceptation. '
        'La droite de tendance permet d\'observer la direction générale '
        'de la relation. '
        'Certaines variables peuvent être affichées en échelle logarithmique '
        'afin d\'améliorer la lisibilité.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Génération des graphiques..."):
        fig_reg = plot_regplots(df)
        st.pyplot(fig_reg)
        plt.close()
# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 3 — RÉGRESSION LINÉAIRE
# ─────────────────────────────────────────────────────────────────────────────

with tab_lin:
    st.markdown('<div class="stab-title">Régression Linéaire</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="stab-desc">Prédit le taux d\'acceptation (valeur continue 0–100 %) '
        '· Pipeline StandardScaler + LinearRegression · Split 80/20 '
        '· Métriques : RMSE et R²</div>',
        unsafe_allow_html=True)

    if show_lin:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE",          f"{lin_m['RMSE']:.4f} pts",
                  help="Plus c'est bas, mieux c'est")
        c2.metric("R² Score",      f"{lin_m['R² Score']:.4f}",
                  help="1.0 = prédiction parfaite")
        c3.metric("Train",         f"{int((1-test_size)*df.shape[0]):,} obs.")
        c4.metric("Test",          f"{int(test_size*df.shape[0]):,} obs.")

        quality = "excellent" if lin_m["R² Score"] > 0.85 else \
                  "bon"       if lin_m["R² Score"] > 0.65 else "perfectible"
        st.markdown(
            f'<div class="ok-box">📌 R² = <b>{lin_m["R² Score"]:.4f}</b> → '
            f'Le modèle explique <b>{lin_m["R² Score"]*100:.1f} %</b> de la variance '
            f'du taux d\'acceptation sur les vraies données ORION. '
            f'Performance : <b>{quality}</b>.<br>'
            f'RMSE = <b>{lin_m["RMSE"]:.2f} pts</b> d\'erreur moyenne.</div>',
            unsafe_allow_html=True
        )

        st.markdown("#### Résidus & Réel vs Prédit")
        st.pyplot(lin_fig)
        plt.close()

        st.markdown("#### Tableau comparatif — 50 premières observations du jeu de test")
        st.dataframe(
            lin_df.style
            .background_gradient(subset=["Erreur abs. (pts)"],
                                 cmap="YlOrRd", vmin=0, vmax=20)
            .format({"Vraie valeur (%)": "{:.2f}",
                     "Prédiction (%)":   "{:.2f}",
                     "Erreur abs. (pts)":"{:.2f}"}),
            use_container_width=True, height=430
        )
    else:
        st.info("Activez la Régression Linéaire dans la sidebar ☑️")


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 4 — RÉGRESSION LOGISTIQUE
# ─────────────────────────────────────────────────────────────────────────────

with tab_log:
    st.markdown('<div class="stab-title">Régression Logistique</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="stab-desc">Prédit la probabilité d\'acceptation '
        '(0 = refusé si taux < 20 %, 1 = accepté si taux ≥ 20 %) '
        '· Pipeline StandardScaler + LogisticRegression(class_weight=balanced) '
        '· Métriques : Accuracy, Precision, Recall, F1</div>',
        unsafe_allow_html=True)

    if show_log:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{log_m['Accuracy']:.4f}")
        c2.metric("Precision", f"{log_m['Precision']:.4f}")
        c3.metric("Recall",    f"{log_m['Recall']:.4f}")
        c4.metric("F1 Score",  f"{log_m['F1 Score']:.4f}")

        st.markdown(
            '<div class="info-box">'
            '<b>Accuracy</b> : taux de bonne classification globale. '
            '<b>Precision</b> : parmi les "acceptés" prédits, combien le sont vraiment. '
            '<b>Recall</b> : parmi les vraiment acceptés, combien ont été détectés. '
            '<b>F1</b> = moyenne harmonique Precision/Recall — '
            'métrique la plus équilibrée quand les classes sont déséquilibrées.</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.markdown("#### Matrice de confusion")
            st.pyplot(log_fig)
            plt.close()
        with col2:
            st.markdown("#### Tableau comparatif — 50 obs. du jeu de test")
            def style_correct(val):
                return ("background-color:#E0F5EF;color:#085041"
                        if val else "background-color:#FDECEA;color:#7A1F1F")
            st.dataframe(
                log_df.style
                .applymap(style_correct, subset=["Correct"])
                .format({"Prob. acceptation": "{:.3f}"}),
                use_container_width=True, height=430
            )
    else:
        st.info("Activez la Régression Logistique dans la sidebar ☑️")


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 5 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

with tab_clf:
    st.markdown('<div class="stab-title">Classification (DT + KNN)</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="stab-desc">Prédit la catégorie du taux : '
        'Faible (<15 %) · Modéré (15–40 %) · Élevé (>40 %) '
        '· Arbre de décision · KNN uniforme · KNN pondéré · KNN normalisé pondéré '
        '· Split 80/20</div>',
        unsafe_allow_html=True)

    if show_clf:
        st.markdown("#### Comparaison Accuracy & F1 — Tous les classifieurs")
        st.pyplot(clf_fig_cmp)
        plt.close()

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("#### Matrice de confusion — Meilleur modèle")
            st.pyplot(clf_fig_cm)
            plt.close()
        with col2:
            st.markdown("#### Métriques détaillées")
            for name, res in clf_r.items():
                icon = {"Arbre de décision":           "🌳",
                        "KNN uniforme (k=7)":          "👥",
                        "KNN pondéré distance (k=7)":  "⚖️",
                        "KNN pondéré normalisé (k=5)": "🎯"}.get(name, "•")
                with st.expander(f"{icon} {name}", expanded=False):
                    m = res["metrics"]
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Accuracy",  f"{m['Accuracy']:.4f}")
                    cc2.metric("Precision", f"{m['Precision']:.4f}")
                    cc3.metric("Recall",    f"{m['Recall']:.4f}")
                    cc4.metric("F1 Score",  f"{m['F1 Score']:.4f}")
                    if st.checkbox(f"Rapport complet", key=f"rep_{name}"):
                        st.code(res["report"])

        st.markdown("#### Techniques d'optimisation")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="info-box">
            <b>⚖️ Pondération des voisins (weights='distance')</b><br>
            Les voisins plus proches reçoivent un poids plus élevé au lieu
            d'un poids uniforme. Améliore les frontières de décision dans
            les espaces à forte hétérogénéité (pib_hab vs nb_cat ont des
            échelles très différentes).
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="info-box">
            <b>📏 Normalisation des variables (StandardScaler)</b><br>
            Centre et réduit chaque feature (μ=0, σ=1). Indispensable pour
            KNN : sans normalisation, pib_hab (~40 000€) domine les distances
            face à nb_cat (0–5), faussant complètement le calcul des voisins.
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Tableau comparatif — Vraies classes vs Prédictions (50 obs.)")
        st.dataframe(clf_df.head(50), use_container_width=True, height=380)
    else:
        st.info("Activez Classification dans la sidebar ☑️")


# ─────────────────────────────────────────────────────────────────────────────
# ONGLET 6 — RÉCAPITULATIF
# ─────────────────────────────────────────────────────────────────────────────

with tab_recap:
    st.markdown('<div class="stab-title">Tableau récapitulatif global</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="stab-desc">Comparaison de tous les modèles — '
        'métriques adaptées à chaque type — vraies données ORION</div>',
        unsafe_allow_html=True)

    summary = build_summary(lin_m, log_m, clf_r)

    def color_summary(df_s):
        styles = pd.DataFrame("", index=df_s.index, columns=df_s.columns)

        for i, row in df_s.iterrows():
            modèle = row["Modèle"]

            if "Linéaire" in modèle:
            # Bleu clair lisible
              styles.loc[i, :] = "background-color:#DCE8FF; color:#0A2357;"
            elif "Logistique" in modèle:
            # Turquoise pastel
              styles.loc[i, :] = "background-color:#DFF7F2; color:#064F43;"
            else:
            # Or pastel pour la classification
              styles.loc[i, :] = "background-color:#FFF4D6; color:#7A4E00;"

        return styles

    st.dataframe(
        summary.style.apply(color_summary, axis=None)
               .set_properties(**{"font-size": "13px", "text-align": "center"}),
        use_container_width=True, height=310
    )

    st.markdown("---")
    st.markdown("#### 🏆 Meilleur modèle par tâche")

    best_clf = max(clf_r, key=lambda n: clf_r[n]["metrics"]["F1 Score"])
    best_clf_m = clf_r[best_clf]["metrics"]

    col1, col2, col3 = st.columns(3)
    for col, title, name, color, vals in [
        (col1, "Régression (continue)", "Régression Linéaire", HCR_BLUE,
         f"R² = {lin_m['R² Score']:.4f}<br>RMSE = {lin_m['RMSE']:.4f} pts"),
        (col2, "Classification binaire","Régression Logistique", HCR_TEAL,
         f"Accuracy = {log_m['Accuracy']:.4f}<br>F1 Score = {log_m['F1 Score']:.4f}"),
        (col3, "Classification 3 classes", best_clf, HCR_GOLD,
         f"Accuracy = {best_clf_m['Accuracy']:.4f}<br>F1 Score = {best_clf_m['F1 Score']:.4f}"),
    ]:
        col.markdown(f"""
        <div style="background:white;border-radius:10px;padding:16px;
                    border-top:4px solid {color};
                    box-shadow:0 1px 4px rgba(10,35,87,.08)">
          <div style="font-size:10.5px;font-weight:700;color:{color};
                      text-transform:uppercase;letter-spacing:.06em;margin-bottom:7px">
          {title}</div>
          <div style="font-size:17px;font-weight:700;color:#0A2357;margin-bottom:6px">
          {name}</div>
          <div style="font-size:12.5px;color:#4A5E82;line-height:1.8">
          {vals}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📌 Pipelines sklearn utilisées")
    st.code("""
# Régression Linéaire
Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

# Régression Logistique
Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=500, class_weight="balanced"))
])

# Arbre de décision
Pipeline([
    ("scaler", StandardScaler()),
    ("model",  DecisionTreeClassifier(max_depth=6, class_weight="balanced"))
])

# KNN pondéré + normalisé (optimisé)
Pipeline([
    ("scaler", StandardScaler()),           # normalisation des variables
    ("model",  KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",                 # pondération des voisins
        metric="euclidean"
    ))
])
    """, language="python")

    st.markdown("""
    <div class="info-box">
    ℹ️ Les modèles sont entraînés sur les <b>vraies données</b> du projet ORION
    (<code>dataset_final.csv</code>) issu de la fusion de 6 sources officielles :
    UNHCR Refugee Data Finder · Eurostat (PIB/hab) · EM-DAT/CRED.
    Les features sont : <code>annee, dec_total, nb_demandes, pib_hab,
    nb_cat, deces, affect</code>.
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;font-size:11px;color:#7B8FB5;padding:6px 0">
  <b style="color:#0A2357">Projet O.R.I.O.N</b> · FANGNON Ornella ·
  Bachelor 2 Data & BI · NEXA Digital School ·
  Sources : UNHCR · Eurostat · EM-DAT/CRED · 2015–2023
</div>
""", unsafe_allow_html=True)

