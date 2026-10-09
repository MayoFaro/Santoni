# Référence baseline — `main` à `7ebcc8a`

Binaire figé de `main` au moment où la branche `improve/tt-memory` en a divergé
(fusion des 55 pouvoirs et de l'Arène à information parfaite). Sert de
baseline pour comparer la variante table de transposition / `Entry` allégée.

- `libsantoni_engine.so` : compilation locale optimisée (`cargo build --release`).
- `source.tar.gz` : `git archive` du commit exact.
- `reference.json` : empreintes et environnement de compilation.

Ne pas modifier ces artefacts pour tester une amélioration ; reconstruire une
nouvelle référence nommée si `main` doit être repris comme baseline à un autre commit.
