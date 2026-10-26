# Données brutes

Fichiers sources lus directement par `notebooks/analyse_hcr.ipynb`, sans extraction préalable. Ne pas les renommer.

| Fichier | Source | Contenu |
|---|---|---|
| `HCR_decision_d_asile.zip` | UNHCR Refugee Data Finder — `asylum_decisions.csv` | Décisions d'asile, 2000–2025 |
| `HCR_demande__d_asile.zip` | UNHCR Refugee Data Finder — `asylum_applications.csv` | Demandes d'asile, 2000–2025 |
| `HCR_chiffre_population.zip` | UNHCR Refugee Data Finder — `persons_of_concern.csv` | Personnes relevant du mandat, 1951–2025 |
| `Eurostat_PIB_par_Habitant_UE.gz` | Eurostat — NAMA_10_PC | PIB par habitant en EUR courants |
| `Eurostat_PIB_UE.gz` | Eurostat — NAMA_10_GDP | PIB total |
| `emdat.xlsx` | EM-DAT / CRED, UCLouvain (export du 14/02/2026) | Catastrophes naturelles et technologiques |

## complementaires/

Sources évaluées dans le rapport mais non retenues, faute d'une couverture temporelle suffisante sur 2015–2023 :

| Fichier | Jeu Eurostat | Contenu |
|---|---|---|
| `Eurostat_Protection_sociale.gz` | SPR_EXP_TYPE | Dépenses de protection sociale (€/hab), à partir de 2018 |
| `Eurostat_Taux_chomage.gz` | LFSA_URGANEDM | Taux de chômage par statut migratoire, à partir de 2021 |
| `Eurostat_Taux_emploi.gz` | LFSA_ERGANEDM | Taux d'emploi par statut migratoire, à partir de 2021 |

Liens : [UNHCR Refugee Data Finder](https://www.unhcr.org/refugee-statistics/) · [Eurostat](https://ec.europa.eu/eurostat/databrowser/) · [EM-DAT](https://www.emdat.be/)
