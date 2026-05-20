# Analyse des taux d'acceptation des demandes d'asile africaines en Europe
**Projet data — HCR, département Data & Analyse stratégique**

---

## Problématique

Pourquoi les taux d'acceptation des demandes d'asile africaines varient-ils autant entre les pays européens ? Sur la période 2015–2023, le taux moyen en Europe est de 19,5 %, mais il va de 0 % en Lettonie à 61 % en Norvège. Ce projet cherche à expliquer ces écarts à partir de données officielles.

Quatre hypothèses sont testées :
- **H1** — les pays qui reçoivent beaucoup de demandes ont des taux plus faibles
- **H2** — les demandeurs venant de pays africains en crise obtiennent des taux plus élevés
- **H3** — les taux varient selon les périodes de pression migratoire
- **H4** — les pays européens plus riches acceptent proportionnellement plus de demandes

---

## Structure du projet

```
rendu_final/
├── analyse_hcr.ipynb       ← notebook principal (tout le code)
├── requetes.sql            ← requêtes SQL commentées
├── requirements.txt        ← librairies à installer
├── README.md               ← ce fichier
├── data/                   ← fichiers sources (ne pas modifier)
│   ├── HCR_decision_d_asile.zip
│   ├── HCR_demande__d_asile.zip
│   ├── HCR_chiffre_population.zip
│   ├── Eurostat_PIB_par_Habitant_UE.gz
│   ├── Eurostat_PIB_UE.gz
│   └── emdat.xlsx
└── output/                 ← généré automatiquement par le notebook
    ├── dataset_final.csv
    ├── decisions_clean.csv
    ├── demandes_clean.csv
    ├── pib_clean.csv
    ├── crises_clean.csv
    ├── hcr.db
    ├── viz1_taux_par_pays.png
    ├── viz2_heatmap.png
    ├── h1_volume_taux.png
    ├── h2_crises_taux.png
    ├── h3_evolution.png
    └── h4_pib_taux.png
```

---

## Installation et lancement

### Étape 1 — Créer un environnement virtuel

```bash
# Windows
python -m venv env
env\Scripts\activate

# Mac / Linux
python3 -m venv env
source env/bin/activate
```

### Étape 2 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 3 — Ouvrir le notebook dans VSCode

1. Ouvrir le dossier `rendu_final/` dans VSCode
2. Ouvrir `analyse_hcr.ipynb`
3. Sélectionner le kernel `env` (environnement virtuel créé à l'étape 1)
4. Cliquer sur **Run All**

Le dossier `output/` est créé automatiquement. Les fichiers CSV et les graphiques s'y retrouvent après exécution.

### Utiliser les requêtes SQL

**Via DB Browser for SQLite** (interface graphique, gratuit) :
1. Télécharger sur https://sqlitebrowser.org/
2. Ouvrir `output/hcr.db`
3. Onglet "Exécuter le SQL" → coller les requêtes depuis `requetes.sql`

**Via Python** :
```python
import sqlite3, pandas as pd
conn = sqlite3.connect('./output/hcr.db')
df = pd.read_sql("SELECT * FROM dataset_final LIMIT 5", conn)
print(df)
conn.close()
```

---

## Sources des données

| Fichier | Source | Contenu |
|---------|--------|---------|
| `HCR_decision_d_asile.zip` | UNHCR Refugee Data Finder | Décisions d'asile 2000–2025 |
| `HCR_demande__d_asile.zip` | UNHCR Refugee Data Finder | Demandes déposées 2000–2025 |
| `HCR_chiffre_population.zip` | UNHCR Refugee Data Finder | Stock de réfugiés 1951–2025 |
| `Eurostat_PIB_par_Habitant_UE.gz` | Eurostat – NAMA_10_PC | PIB par habitant en EUR courants |
| `Eurostat_PIB_UE.gz` | Eurostat – NAMA_10_GDP | PIB total en millions EUR |
| `emdat.xlsx` | EM-DAT / CRED (UCLouvain) | Catastrophes naturelles et technologiques |

---

## Périmètre de l'analyse

| Paramètre | Valeur |
|-----------|--------|
| Période | 2015–2023 |
| Pays d'accueil | EU27 + Norvège + Suisse (29 pays) |
| Pays d'origine | 54 pays africains |
| Niveau de décision | Première instance (FI), procédure générale (G) |
| Variable cible | Taux d'acceptation = décisions positives / total × 100 |
| Lignes dans le dataset final | 4 867 |

---

## Contenu du notebook

| Section | Contenu |
|---------|---------|
| 1. Imports | Librairies Python |
| 2. Paramètres | Listes de pays, correspondances ISO |
| 3. Chargement | Lecture des ZIP, GZ et Excel |
| 4. Exploration | Aperçu, valeurs manquantes, distributions |
| 5. Nettoyage | Filtres, renommage, calcul du taux |
| 6. SQL | 5 requêtes SQLite : jointures, agrégations, filtres |
| 7. Fusion | Construction du dataset final |
| 8. Export | CSV + base SQLite |
| 9. Visualisations | Barplot + heatmap exploratoires |
| 10. Hypothèses | Test H1 à H4 avec corrélations et graphiques |
| 11. Synthèse | Tableau récapitulatif des 4 verdicts |
| 12. Conclusions | Réponse à la problématique, limites, pistes |

---

## Résultats principaux

| Hypothèse | Verdict | Statistique |
|-----------|---------|-------------|
| H1 — Volume → taux plus faible | ❌ Infirmée | r = 0.022, non significatif |
| H2 — Pays en crise → taux plus élevé | ⚠️ Partielle | r = -0.016 (EM-DAT ≠ conflits armés) |
| H3 — Effet temporel | ✅ Confirmée | Écart de 8.2 pts sur la période |
| H4 — PIB élevé → taux plus élevé | ✅ Confirmée | r = 0.754, p < 0.001 |

Le facteur économique est le plus déterminant : les pays à PIB élevé acceptent en moyenne 40 % des demandes contre 9 % pour les pays à faible PIB.
