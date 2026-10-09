# Référence baseline — `main` à `b698745`

Binaire figé de `main` au moment où la branche `improve/hermes-generation` en
a divergé (juste après la fusion du profilage du clonage de PV et la
clôture du sous-axe « réduction des copies »). Sert d'original figé pour la
campagne appariée de validation de la génération progressive (`visit_base`)
et de la canonisation du swap des deux bâtisseurs (`canonical_key`), voir
`docs/profilage-hermes-20261009.md` et
`tools/hermes_canonicalisation_campaign.py`.

- `libsantoni_engine.so` : compilation locale optimisée (`cargo build --release`).
- `source.tar.gz` : `git archive` du commit exact (dépôt entier, pas
  seulement `native_engine`, pour pouvoir charger le paquet `santorini`
  figé dans un module Python séparé lors de la comparaison).
- `reference.json` : empreintes et environnement de compilation.

Ne pas modifier ces artefacts pour tester une amélioration ; reconstruire une
nouvelle référence nommée si `main` doit être repris comme baseline à un
autre commit.
