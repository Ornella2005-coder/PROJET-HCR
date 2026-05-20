-- ============================================================
-- PROJET O.R.I.O.N — ANALYSES SQL (H1 à H4)
-- Base : output/hcr.db (SQLite)
-- ============================================================


-- ============================================================
-- 1. TAUX D’ACCEPTATION MOYEN PAR PAYS D’ACCUEIL
-- ============================================================
SELECT pays_accueil,
       SUM(dec_pos)   AS decisions_positives,
       SUM(dec_total) AS decisions_total,
       ROUND(SUM(dec_pos) * 100.0 / SUM(dec_total), 2) AS taux_pct
FROM decisions
GROUP BY pays_accueil
ORDER BY taux_pct DESC;


-- ============================================================
-- 2. ÉVOLUTION DU TAUX D’ACCEPTATION PAR ANNÉE
-- ============================================================
SELECT annee,
       SUM(dec_total) AS nb_decisions,
       ROUND(SUM(dec_pos) * 100.0 / SUM(dec_total), 2) AS taux_pct
FROM decisions
GROUP BY annee
ORDER BY annee;


-- ============================================================
-- 3. TAUX D’ACCEPTATION PAR PAYS × ANNÉE
-- ============================================================
SELECT annee,
       pays_accueil,
       ROUND(SUM(dec_pos) * 100.0 / SUM(dec_total), 2) AS taux_pct
FROM decisions
GROUP BY annee, pays_accueil
ORDER BY annee, taux_pct DESC;


-- ============================================================
-- 4. TOP PAYS AFRICAINS PAR TAUX (min 500 décisions)
-- ============================================================
SELECT pays_origine,
       SUM(dec_total) AS nb_decisions,
       ROUND(SUM(dec_pos) * 100.0 / SUM(dec_total), 2) AS taux_pct
FROM decisions
GROUP BY pays_origine
HAVING SUM(dec_total) > 500
ORDER BY taux_pct DESC
LIMIT 15;


-- ============================================================
-- 5. HYPOTHÈSE H4 — TAUX × PIB PAR HABITANT
-- ============================================================
SELECT d.pays_accueil,
       ROUND(SUM(d.dec_pos) * 100.0 / SUM(d.dec_total), 2) AS taux_pct,
       ROUND(AVG(p.pib_hab), 0) AS pib_moyen
FROM decisions d
LEFT JOIN pib p 
       ON d.iso2_accueil = p.iso2_accueil
      AND d.annee = p.annee
WHERE p.pib_hab IS NOT NULL
GROUP BY d.pays_accueil
ORDER BY pib_moyen DESC;


-- ============================================================
-- 6. HYPOTHÈSE H2 — TAUX × CRISES / CATASTROPHES
-- ============================================================
SELECT d.pays_origine,
       ROUND(SUM(d.dec_pos) * 100.0 / SUM(d.dec_total), 2) AS taux_pct,
       SUM(d.dec_total) AS nb_decisions,
       ROUND(AVG(COALESCE(c.nb_cat, 0)), 1) AS moy_catastrophes
FROM decisions d
LEFT JOIN crises c 
       ON d.iso3_origine = c.iso3_origine
      AND d.annee = c.annee
GROUP BY d.pays_origine
HAVING SUM(d.dec_total) > 500
ORDER BY taux_pct DESC;


-- ============================================================
-- 7. HYPOTHÈSE H1 — TAUX × PRESSION ADMINISTRATIVE (DEMANDES)
-- ============================================================
SELECT d.pays_accueil,
       SUM(a.nb_demandes) AS total_demandes,
       ROUND(SUM(d.dec_pos) * 100.0 / SUM(d.dec_total), 2) AS taux_pct
FROM demandes a
LEFT JOIN decisions d 
       ON a.annee = d.annee
      AND a.iso3_accueil = d.iso3_accueil
GROUP BY d.pays_accueil
HAVING SUM(d.dec_total) > 0
ORDER BY total_demandes DESC;


-- ============================================================
-- 8. DATASET FINAL COMPLET (pour export / visualisation)
-- ============================================================
SELECT *
FROM dataset_final
ORDER BY annee, pays_accueil, pays_origine;



