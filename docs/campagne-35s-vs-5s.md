# Protocole validé : moteur initial, 35 secondes contre 5 secondes

## Organisation

Branche : `experiment/baseline-35s-vs-5s`. Aucune modification du moteur.
Référence figée dans `experiments/references/initial-cb732bb/`.
Les futures améliorations auront chacune leur branche et seront comparées à
cette référence, puis au `main` courant si d'autres améliorations ont déjà été
intégrées. Fusion dans `main` seulement après campagne et validation.

## Attribution expérimentale

Les 100 configurations exactes de `selfplay-100-5s-20261005/campaign.json`
sont réutilisées : placements, pouvoirs, premier joueur et identifiant d'origine.
Chaque configuration donne deux parties : budgets [35, 5], puis [5, 35].
Total : 200 parties, 25 groupes de placements. Les règles, limite de 150 tours,
arrêt sur preuve et validation indépendante des coups restent ceux de l'origine.

Chaque décision enregistre son budget, temps réel, profondeur terminée,
profondeur tentée, score, statut de preuve et variante. Chaque tour est enregistré
atomiquement. Une reprise conserve les parties terminées et archive les tentatives
incomplètes avant de les recommencer. Un verrou empêche deux campagnes simultanées
dans le même dossier. Les empreintes et paramètres doivent correspondre à la reprise.

## Écart matériel documenté

Le PC actuel possède un Intel i5-12500, six cœurs physiques. Six processus sont
épinglés à des cœurs physiques distincts, contre douze lors de la première campagne.
Le binaire historique est absent ; les sources identiques sont recompilées localement.
La comparaison interne 35 s / 5 s utilise le même matériel et le même binaire.
Une comparaison directe des temps et coups avec la campagne historique devra tenir
compte de ces différences. Les décisions à budget mural ne sont pas déterministes.

## Exécution

Depuis la racine du projet, après restauration de `source/` si nécessaire :

```bash
python3 tools/time_budget_campaign.py --output experiments/baseline-35s-vs-5s-20261005 --workers 6
```

Ajouter `--prepare-only` pour vérifier la référence et écrire le manifeste sans
jouer. Au moins 200 Mio doivent être libres sur le disque de sortie au démarrage.
Conserver le PC allumé et éviter des charges CPU concurrentes. La durée sera de
plusieurs heures ; l'estimation historique de 16,9 minutes ne s'applique pas à ce PC.

## Bilan automatique

Le banc produit `summary.json` après chaque partie. À la fin, il réanalyse
séparément les positions 16/26 et 18/28 à 5 s et 35 s, audite les historiques et
contradictions de preuves avec l'outil figé, puis écrit `analysis.json` et `rapport.md`.
Les anciens résultats ne sont jamais réécrits.

Le bilan distingue les victoires par budget, les confrontations de pouvoirs,
les résultats appariés et les profondeurs/temps de recherche. L'intervalle bootstrap
rééchantillonne les 25 groupes de placements entiers, sans traiter les variantes
comme indépendantes. Les profondeurs agrégées ne comparent pas des positions identiques.
Les scores ne sont pas des probabilités de victoire ; une estimation à 35 s ne
devient pas une preuve. Les erreurs et parties non conclues sont signalées.
