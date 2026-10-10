"""Génère un corpus de placements 100 % sous-jeu sans pouvoir pour la campagne
`tools/mcts_az_vs_reference_campaign.py`.

Pourquoi un nouveau corpus : le corpus historique
`experiments/selfplay-100-5s-20261005/campaign.json` est à 88 % composé de
jobs avec pouvoirs (seulement 12 de ses 100 jobs ont `powers == [0, 0]`). Or
le mode `--single-move` du worker candidat `assert!(is_in_scope(&state))` et
panique sur toute position hors du sous-jeu sans pouvoir : une campagne de
200 parties sur ce corpus s'effondre en parties `'error'` et l'IC bootstrap
(conditionné à `len(finished) == 200`) n'est jamais calculé. Aucun corpus
existant du dépôt n'a de jeu sans pouvoir utilisable à cette échelle, d'où ce
générateur.

Structure : mirroir exact du motif de `tools/selfplay.py::jobs` pour la paire
de base `(0, 0)` — « 25 placements déclinés en quatre variantes de premier
joueur et d'attribution des pouvoirs ; les variantes sans pouvoirs
comprenaient une rotation du plateau » (`docs/ameliorations-moteur-20261005.md`
§ campagne initiale) — simplement mis à l'échelle de 3 à 25 ouvertures
puisque *toutes* les ouvertures sont ici sans pouvoir :

    25 ouvertures × 2 orientations (plateau brut / rotation 180°) × 2 premiers
    joueurs = 100 jobs, 50 placements distincts, `powers == (0, 0)` partout.

La clé de groupe de la campagne est `(tuple(matchup), opening)` : avec un
`matchup` constant `[0, 0]`, les 25 ouvertures donnent 25 groupes de 4 jobs,
soit 8 parties par groupe après croisement des deux couleurs (200 parties) —
exactement la granularité de bootstrap par groupe de placement des campagnes
précédentes, et la même que celle des 3 groupes sans pouvoir du corpus
historique.

Graine : 20261010, distincte de la graine 20261005 du corpus historique, pour
que les placements de cette campagne ne soient pas les mêmes tirages (et
restent reproductibles : relancer ce script redonne bit pour bit le même
fichier, au `started_at` près).

Validation : chaque job est passé à `santorini.engine.validate_setup` avant
écriture, et `powers == (0, 0)` est réaffirmé job par job. Toute anomalie
lève — on ne commit jamais un corpus invalide.

Usage : `python3 tools/generate_no_power_corpus.py` (écrit
`experiments/no-power-corpus-20261010/campaign.json`), ou
`--out <chemin>` pour un essai hors arborescence committée.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'experiments/no-power-corpus-20261010/campaign.json'
SEED = 20261010
OPENINGS = 25
MATCHUP = (0, 0)
POWERS = (0, 0)

sys.path.insert(0, str(ROOT))
from santorini.engine import validate_setup  # noqa: E402
from santorini import native  # noqa: E402


def atomic_json(path, value):
    """Écriture atomique (temporaire + renommage), même fonction que
    `tools/hermes_canonicalisation_campaign.py::atomic_json` et que
    `tools/selfplay.py` : aucun fichier partiellement écrit ne doit jamais
    être visible sous le nom final."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def rotate(cells):
    """Rotation 180° du plateau 5x5 (case c -> 24-c), la même transformation
    que `tools/selfplay.py` applique à la seconde variante de la paire sans
    pouvoir (`board = [24-c for c in cells] if pair == (0, 0) and reverse`)."""
    return [24 - c for c in cells]


def distinct_after_rotation(cells):
    """Vrai si la rotation 180° donne un placement réellement différent.

    Un tirage symétrique par rotation (ex. `[0, 24, ...]` : {0,24} == {24,0})
    produirait deux jobs identiques au lieu de deux placements distincts dans
    le même groupe. `tools/selfplay.py` ne s'en préoccupait pas (3 ouvertures
    sans pouvoir seulement) ; à 25 ouvertures on retire ces tirages pour
    garantir 50 placements distincts.
    """
    turned = rotate(cells)
    return {cells[0], cells[1]} != {turned[0], turned[1]} or \
           {cells[2], cells[3]} != {turned[2], turned[3]}


def build_jobs(seed=SEED, openings=OPENINGS):
    rng = random.Random(seed)
    result = []
    for opening in range(openings):
        while True:
            cells = rng.sample(range(25), 4)  # 4 cases distinctes parmi 25
            if distinct_after_rotation(cells):
                break
        for reverse in (False, True):
            board = rotate(cells) if reverse else cells
            for first in (0, 1):
                result.append(dict(number=len(result) + 1, matchup=list(MATCHUP),
                                   opening=opening, powers=list(POWERS),
                                   workers=[board[:2], board[2:]], first=first))
    return result


def validate(jobs):
    """Rejette fort et tôt plutôt que d'écrire un corpus douteux : chaque job
    doit être une position de départ légale ET strictement sans pouvoir."""
    seen = set()
    for job in jobs:
        assert tuple(job['powers']) == POWERS, \
            f"job {job['number']}: powers {job['powers']} != {list(POWERS)}"
        assert tuple(job['matchup']) == MATCHUP, f"job {job['number']}: matchup inattendu"
        cells = [c for pair in job['workers'] for c in pair]
        assert len(set(cells)) == 4, f"job {job['number']}: bâtisseurs superposés {cells}"
        assert all(0 <= c <= 24 for c in cells), f"job {job['number']}: case hors plateau {cells}"
        assert job['first'] in (0, 1), f"job {job['number']}: premier joueur invalide"
        # La validation de référence : le moteur de règles lui-même.
        validate_setup(job['workers'], job['powers'], job['first'])
        seen.add((tuple(map(tuple, job['workers'])), job['first']))
    assert len(seen) == len(jobs), 'jobs dupliqués (placement + premier joueur)'
    placements = {tuple(map(tuple, job['workers'])) for job in jobs}
    assert len(placements) == len(jobs) // 2, \
        f'{len(placements)} placements distincts pour {len(jobs)} jobs (2 attendus par placement)'


def engine_digest():
    """Provenance : sha256 de la bibliothèque moteur présente au moment de la
    génération. `validate_setup` du sous-jeu sans pouvoir est du Python pur
    (aucun pouvoir n'exige le moteur natif), donc ce champ est de la
    provenance, pas une dépendance de la génération — il garde simplement le
    schéma du corpus historique rempli de façon vérifiable."""
    path = native.library_path()
    if not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--openings', type=int, default=OPENINGS)
    args = parser.parse_args()

    jobs = build_jobs(args.seed, args.openings)
    validate(jobs)
    corpus = {
        'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        # Aucune recherche n'est exécutée ici : ce fichier est un corpus de
        # placements, pas le journal d'une campagne jouée. Le budget de temps
        # par tour est choisi par le script consommateur (`--seconds`).
        'seconds': None,
        'workers': 1,
        'cpus': [],
        'seed': args.seed,
        'jobs': jobs,
        'engine_sha256': engine_digest(),
        'notes': ('No-power corpus for tools/mcts_az_vs_reference_campaign.py: '
                  f'{args.openings} openings x 2 board orientations (raw / 180 deg rotation) '
                  f'x 2 first players = {len(jobs)} jobs, every job powers == [0, 0]. '
                  'Same schema and same placement-group pattern as '
                  'experiments/selfplay-100-5s-20261005/campaign.json (see '
                  'tools/selfplay.py::jobs), scaled from 3 to '
                  f'{args.openings} base-game openings. Generated by '
                  f'tools/generate_no_power_corpus.py with seed {args.seed}; every job '
                  'validated by santorini.engine.validate_setup. No game was played '
                  'to produce this file.'),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, corpus)
    print(f'{out}: {len(jobs)} jobs, '
          f"{len({tuple(map(tuple, j['workers'])) for j in jobs})} placements distincts, "
          f"{len({(tuple(j['matchup']), j['opening']) for j in jobs})} groupes de placement, "
          'tous sans pouvoir.')


if __name__ == '__main__':
    main()
