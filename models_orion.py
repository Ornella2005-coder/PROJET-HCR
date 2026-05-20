"""
PROJET O.R.I.O.N — Pipeline ML — Vraies données
HCR / UNHCR — Bureau Europe
Auteure : FANGNON Ornella — Bachelor 2 Data & BI — NEXA Digital School

Colonnes du dataset_final.csv (généré par analyse_hcr.ipynb) :
    annee, pays_accueil, iso3_accueil, iso2_accueil,
    pays_origine, iso3_origine,
    dec_pos, dec_rej, dec_total, taux,
    nb_demandes, pib_hab, nb_cat, deces, affect

Variables retenues pour le ML :
    Features  : annee, dec_total, nb_demandes, pib_hab, nb_cat, deces, affect
    Target    : taux            (régression linéaire)
                accepte_binaire (régression logistique — 0/1)
                categorie       (classification — Faible / Modéré / Élevé)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

#Couleurs HCR 
HCR_NAVY = "#0A2357"
HCR_BLUE = "#1A4A9B"
HCR_SKY  = "#2E82D4"
HCR_GOLD = "#F5A800"
HCR_TEAL = "#00A878"
HCR_RED  = "#D94035"

# Colonnes
FEATURES_NUM = ["annee", "dec_total", "nb_demandes", "pib_hab",
                "nb_cat", "deces", "affect"]
TARGET_REG   = "taux"
TARGET_BIN   = "accepte_binaire"
TARGET_CAT   = "categorie"

# Chemin par défaut (même dossier que ce fichier → output/dataset_final.csv)
_HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(_HERE, "output", "dataset_final.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES VRAIES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_data(path=None):
    """
    Charge output/dataset_final.csv généré par analyse_hcr.ipynb.
    path : chemin personnalisé (optionnel, pour Streamlit file_uploader).
    """
    p = path or DATA_PATH

    if isinstance(p, str):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Fichier introuvable : {p}\n"
                "→ Lancez d'abord analyse_hcr.ipynb pour générer output/dataset_final.csv"
            )
        df = pd.read_csv(p)
    else:
        # objet fichier (BytesIO) envoyé par st.file_uploader
        df = pd.read_csv(p)

    required = ["annee", "dec_pos", "dec_total", "taux",
                "nb_demandes", "pib_hab", "nb_cat", "deces", "affect"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    df = df.dropna(subset=FEATURES_NUM + [TARGET_REG]).reset_index(drop=True)

    # Variables cibles dérivées
    df[TARGET_BIN] = (df[TARGET_REG] >= 20).astype(int)
    df[TARGET_CAT] = pd.cut(
        df[TARGET_REG],
        bins=[-0.1, 15, 40, 100],
        labels=["Faible", "Modéré", "Élevé"]
    ).astype(str)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CORRÉLATIONS & REGPLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(df):
    corr = df[FEATURES_NUM + [TARGET_REG]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f",
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        "Matrice de corrélation — Features vs Taux d'acceptation",
        fontsize=11, color=HCR_NAVY, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    return fig


def plot_regplots(df):
    palette = [HCR_NAVY, HCR_TEAL, HCR_SKY, HCR_GOLD, HCR_RED, HCR_BLUE, "#7F77DD"]
    ncols, n = 4, len(FEATURES_NUM)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    hypo_map = {
        "annee":       "H3 Temporel",
        "dec_total":   "Volume décisions",
        "nb_demandes": "H1 Volume ❌",
        "pib_hab":     "H4 PIB ✅",
        "nb_cat":      "H2 Crise ⚠️",
        "deces":       "H2 Décès",
        "affect":      "H2 Affectés",
    }

    for i, (feat, col) in enumerate(zip(FEATURES_NUM, palette)):
        use_log = (df[feat].max() / (df[feat].min() + 1)) > 1000
        sns.regplot(
            data=df, x=feat, y=TARGET_REG, ax=axes[i],
            scatter_kws={"alpha": 0.20, "s": 10, "color": col},
            line_kws={"color": HCR_NAVY, "linewidth": 2},
            ci=95, logx=use_log,
        )
        r = df[[feat, TARGET_REG]].corr().iloc[0, 1]
        axes[i].set_title(
            f"{feat}\nr = {r:.3f}  [{hypo_map.get(feat, '')}]",
            fontsize=9, color=HCR_NAVY, fontweight="bold",
        )
        axes[i].set_xlabel(feat, fontsize=8)
        axes[i].set_ylabel("Taux (%)" if i % ncols == 0 else "", fontsize=8)
        axes[i].tick_params(labelsize=7)
        for sp in axes[i].spines.values():
            sp.set_edgecolor("#DDE3EF")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Regplots — Features vs Taux d'acceptation (dataset_final.csv)",
        fontsize=13, color=HCR_NAVY, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SPLIT 80 / 20
# ═══════════════════════════════════════════════════════════════════════════════

def split_data(df, target, test_size=0.2, random_state=42):
    X = df[FEATURES_NUM]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PIPELINES SKLEARN
# ═══════════════════════════════════════════════════════════════════════════════

def build_linear_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])


def build_logistic_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=500, random_state=42,
                                      class_weight="balanced")),
    ])


def build_dt_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  DecisionTreeClassifier(max_depth=6, random_state=42,
                                          class_weight="balanced")),
    ])


def build_knn_pipeline(k=7, weights="uniform"):
    return Pipeline([
        ("scaler", StandardScaler()),       
        ("model",  KNeighborsClassifier(
            n_neighbors=k,
            weights=weights,                
            metric="euclidean",
        )),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENTRAÎNEMENT & ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_linear(df):
    X_tr, X_te, y_tr, y_te = split_data(df, TARGET_REG)
    pipe = build_linear_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    r2   = float(r2_score(y_te, y_pred))

    df_cmp = pd.DataFrame({
        "Vraie valeur (%)":  np.round(y_te.values[:50], 2),
        "Prédiction (%)":    np.round(y_pred[:50], 2),
        "Erreur abs. (pts)": np.round(np.abs(y_te.values[:50] - y_pred[:50]), 2),
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, y_te.values - y_pred,
                    alpha=0.3, color=HCR_SKY, s=8, edgecolors="none")
    axes[0].axhline(0, color=HCR_RED, linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Valeurs prédites (%)", fontsize=10)
    axes[0].set_ylabel("Résidus (pts)", fontsize=10)
    axes[0].set_title("Résidus — Régression Linéaire",
                       fontsize=11, color=HCR_NAVY, fontweight="bold")

    mn, mx = float(y_te.min()), float(y_te.max())
    axes[1].scatter(y_te.values, y_pred, alpha=0.3, color=HCR_TEAL,
                    s=8, edgecolors="none")
    axes[1].plot([mn, mx], [mn, mx], color=HCR_RED, linewidth=1.8,
                 linestyle="--", label="Prédiction parfaite")
    axes[1].set_xlabel("Taux réel (%)", fontsize=10)
    axes[1].set_ylabel("Taux prédit (%)", fontsize=10)
    axes[1].set_title(f"Réel vs Prédit — R² = {r2:.4f}",
                       fontsize=11, color=HCR_NAVY, fontweight="bold")
    axes[1].legend(fontsize=9)
    fig.tight_layout()

    return pipe, {"RMSE": round(rmse, 4), "R² Score": round(r2, 4)}, df_cmp, fig, y_te, y_pred


def train_logistic(df):
    X_tr, X_te, y_tr, y_te = split_data(df, TARGET_BIN)
    pipe = build_logistic_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred      = pipe.predict(X_te)
    y_pred_prob = pipe.predict_proba(X_te)[:, 1]

    metrics = {
        "Accuracy":  round(float(accuracy_score(y_te, y_pred)), 4),
        "Precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        "F1 Score":  round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
    }

    df_cmp = pd.DataFrame({
        "Vraie classe":       y_te.values[:50],
        "Classe prédite":     y_pred[:50],
        "Prob. acceptation":  np.round(y_pred_prob[:50], 3),
        "Correct":            (y_te.values[:50] == y_pred[:50]),
    })

    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d",
                cmap=sns.light_palette(HCR_BLUE, as_cmap=True),
                ax=ax,
                xticklabels=["Refusé (0)", "Accepté (1)"],
                yticklabels=["Refusé (0)", "Accepté (1)"],
                linewidths=0.5, cbar=False)
    ax.set_xlabel("Classe prédite", fontsize=10)
    ax.set_ylabel("Classe réelle", fontsize=10)
    ax.set_title(
        f"Matrice de confusion — Régression Logistique\n"
        f"Accuracy = {metrics['Accuracy']:.4f}  |  F1 = {metrics['F1 Score']:.4f}",
        fontsize=10, color=HCR_NAVY, fontweight="bold")
    fig.tight_layout()

    return pipe, metrics, df_cmp, fig, y_te, y_pred


def train_classification(df):
    X_tr, X_te, y_tr, y_te = split_data(df, TARGET_CAT)

    models = {
        "Arbre de décision":           build_dt_pipeline(),
        "KNN uniforme (k=7)":          build_knn_pipeline(7, "uniform"),
        "KNN pondéré distance (k=7)":  build_knn_pipeline(7, "distance"),
        "KNN pondéré normalisé (k=5)": build_knn_pipeline(5, "distance"),
    }

    results = {}
    df_cmp  = pd.DataFrame({"Vraie classe": y_te.values[:50]})

    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        results[name] = {
            "pipe":    pipe,
            "y_pred":  y_pred,
            "metrics": {
                "Accuracy":  round(float(accuracy_score(y_te, y_pred)), 4),
                "Precision": round(float(precision_score(y_te, y_pred,
                                    average="weighted", zero_division=0)), 4),
                "Recall":    round(float(recall_score(y_te, y_pred,
                                    average="weighted", zero_division=0)), 4),
                "F1 Score":  round(float(f1_score(y_te, y_pred,
                                    average="weighted", zero_division=0)), 4),
            },
            "report": classification_report(y_te, y_pred, zero_division=0),
        }
        df_cmp[f"Préd. {name}"] = y_pred[:50]

    # Figure comparaison
    names = list(results.keys())
    accs  = [results[n]["metrics"]["Accuracy"]  for n in names]
    f1s   = [results[n]["metrics"]["F1 Score"]  for n in names]
    x, w  = np.arange(len(names)), 0.35

    fig_cmp, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, accs, w, label="Accuracy", color=HCR_BLUE, alpha=0.85)
    ax.bar(x + w/2, f1s,  w, label="F1 Score", color=HCR_GOLD, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Comparaison Accuracy & F1 — Classifieurs",
                  fontsize=11, color=HCR_NAVY, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    fig_cmp.tight_layout()

    # Meilleur modèle — matrice de confusion
    best    = max(results, key=lambda n: results[n]["metrics"]["F1 Score"])
    classes = sorted(y_te.unique())
    cm_b    = confusion_matrix(y_te, results[best]["y_pred"], labels=classes)
    fig_cm, ax2 = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_b, annot=True, fmt="d",
                cmap=sns.light_palette(HCR_TEAL, as_cmap=True),
                ax=ax2, xticklabels=classes, yticklabels=classes,
                linewidths=0.5, cbar=False)
    ax2.set_xlabel("Classe prédite", fontsize=10)
    ax2.set_ylabel("Classe réelle", fontsize=10)
    ax2.set_title(f"Matrice de confusion — {best}",
                   fontsize=10, color=HCR_NAVY, fontweight="bold")
    fig_cm.tight_layout()

    return results, df_cmp, fig_cmp, fig_cm, y_te


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TABLEAU RÉCAPITULATIF
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary(lin_m, log_m, clf_r):
    rows = [
        {"Modèle": "Régression Linéaire",
         "Type": "Régression (valeur continue)",
         "RMSE": lin_m["RMSE"], "R²": lin_m["R² Score"],
         "Accuracy": "—", "Precision": "—", "Recall": "—", "F1 Score": "—"},
        {"Modèle": "Régression Logistique",
         "Type": "Classification binaire (0/1)",
         "RMSE": "—", "R²": "—",
         "Accuracy":  log_m["Accuracy"],  "Precision": log_m["Precision"],
         "Recall":    log_m["Recall"],    "F1 Score":  log_m["F1 Score"]},
    ]
    for name, res in clf_r.items():
        rows.append({
            "Modèle": name, "Type": "Classification 3 classes",
            "RMSE": "—", "R²": "—",
            "Accuracy":  res["metrics"]["Accuracy"],
            "Precision": res["metrics"]["Precision"],
            "Recall":    res["metrics"]["Recall"],
            "F1 Score":  res["metrics"]["F1 Score"],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TEST EN LIGNE DE COMMANDE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    print("=" * 62)
    print("  PROJET O.R.I.O.N — Pipeline ML (vraies données)")
    print("=" * 62)
    df = load_real_data(p)
    print(f"\nDataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"Features : {FEATURES_NUM}")

    _, lin_m, _, _, _, _ = train_linear(df)
    _, log_m, _, _, _, _ = train_logistic(df)
    clf_r, _, _, _, _    = train_classification(df)

    print("\n" + "=" * 62)
    print(build_summary(lin_m, log_m, clf_r).to_string(index=False))
