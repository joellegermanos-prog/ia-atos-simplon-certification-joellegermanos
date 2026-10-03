# Synthèse de détection de dérive — M6-B1

Référence : `reference_set.csv` (500 lignes) · Production : `prod_3months.csv` (3000 lignes)
Features testées : 10 (5 numériques, 5 catégorielles) — exhaustif.
Seuils PSI (repères conventionnels, pas des frontières) : < 0.1 stable · 0.1-0.25 suspect · > 0.25 dérive.
Tests multiples corrigés par Holm-Bonferroni à alpha = 0.05.
Calculs : `src/drift_detection.py`, couvert par `tests/`.

| feature | type | PSI | p-value | signif. (Holm) | verdict | commentaire |
|---|---|---|---|---|---|---|
| usager_id | categorielle | 13.367 | 1.00e+00 | False | dérive | signaux concordants, modalités redistribuées |
| code_insee_commune | categorielle | 8.133 | 1.00e+00 | False | dérive | signaux concordants, modalités redistribuées |
| synthese_entretien | categorielle | 0.292 | 2.80e-02 | False | dérive | signaux concordants, modalités redistribuées |
| code_rome_vise | categorielle | 0.079 | 9.80e-01 | False | stable | aucun signal |
| niveau_diplome | categorielle | 0.023 | 2.59e-02 | False | stable | aucun signal |
| age | numerique | 0.021 | 1.34e-01 | False | stable | aucun signal |
| anciennete_poste_ans | numerique | 0.015 | 7.85e-01 | False | stable | aucun signal |
| classe_retour_emploi | numerique | 0.0 | 1.00e+00 | False | stable | aucun signal |
| est_allocataire | numerique | 0.0 | 1.38e-01 | False | stable | aucun signal |
| nationalite_hors_ue | numerique | 0.0 | 1.00e+00 | False | stable | aucun signal |

## Synthèse globale

3 feature(s) au-delà de PSI 0.25, 0 en zone suspecte.
Features les plus déplacées : `usager_id`, `code_insee_commune`, `synthese_entretien`.

_(2-3 lignes d'interprétation métier à rédiger ici.)_

> Un chiffre n'est pas un verdict, un verdict n'est pas un diagnostic.
