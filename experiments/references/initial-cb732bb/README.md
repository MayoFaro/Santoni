# Référence initiale immuable

Sources moteur : commit `cb732bb`. Archive du dépôt : `52d097c`, qui ajoute
les outils et résultats de campagne sans modifier le moteur ni ses règles.

- `source.tar.gz` : sources Rust, règles et pont Python, outils, tests et documentation.
- `libsantoni_engine.so` : compilation locale optimisée, identique pour les deux budgets.
- `reference.json` : empreintes, commits et environnement de compilation.
- `source/` : extraction locale ignorée par Git ; utilisée pour importer les règles figées.

Le binaire original de la première campagne n'était pas présent sur ce PC.
Son empreinte historique est conservée. Le nouveau binaire diffère : cette
référence est une reconstruction des mêmes sources, pas une copie du binaire historique.
Ne jamais remplacer ces artefacts pour tester une amélioration.

Pour restaurer l'extraction après clonage, depuis la racine :

```bash
mkdir -p experiments/references/initial-cb732bb/source
tar -xzf experiments/references/initial-cb732bb/source.tar.gz -C experiments/references/initial-cb732bb/source
```

Le banc vérifie les empreintes du binaire, de l'archive et des fichiers sources
avant de démarrer. Une modification de référence impose une autre campagne.
