"""Produce a reviewable French report from the completed campaign."""
import json
from pathlib import Path
import statistics
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.analyse_selfplay import analyse
from tools.selfplay import summary
from santorini.engine import Action, Position, Turn


def report(directory):
    p=Path(directory);a=analyse(p);s=summary(p)
    assert a['completed']==100 and a['audit_errors']==[] and a['proof_contradictions']==[], 'Campaign incomplete or audit failed'
    games=[json.loads(f.read_text()) for f in sorted(p.glob('game-*.json'))]
    config=json.loads((p/'campaign.json').read_text())
    wall=(max(datetime.fromisoformat(g['finished_at']) for g in games)-datetime.fromisoformat(config['started_at'])).total_seconds()
    decisions=[d for g in games for d in g['decisions']]
    lines=['# Campagne Santoni : 100 parties robot contre robot', '',
           f"100 parties terminées ; **{a['turns']} tours joués**, {a['decisions']} recherches. Budget maximal : **5 secondes par tour**.",
           f"Durée de la campagne : {wall/60:.1f} minutes, avec {config['workers']} processus à priorité réduite sur des cœurs distincts.", '',
           '## Protocole', '',
           'Même version du moteur Rust des deux côtés, issue de main `cb732bb`. Aucune modification du moteur durant la campagne. Chaque coup publié est validé par les règles Python ; chaque historique est ensuite rejoué intégralement.',
           '25 placements initiaux tirés avec une graine fixe, chacun décliné en quatre parties : inversion du premier joueur et de l’attribution des pouvoirs. Pour les parties sans pouvoirs, la deuxième paire utilise une rotation de 180° du plateau. Les placements sont variés, pas optimisés par le conseiller de placement. Les deux camps reçoivent le même budget.',
           'Les parties de contrôle et les réanalyses à 20 ou 60 secondes décrites plus bas sont exclues du total de 100.',
           f"Graine : `{config['seed']}`. Empreinte de la bibliothèque : `{config['engine_sha256']}`.", '',
           '## Résultats', '',
           '| Pouvoirs | Parties | Victoires | Victoires du premier joueur |',
           '|---|---:|---|---:|']
    for match,v in s['by_matchup'].items():
        wins=', '.join(f'{power} : {count}' for power,count in v['wins_by_power'].items())
        lines.append(f"| {match} | {v['games']} | {wins} | {v['first_player_wins']}/{v['games']} |")
    lines+=['', f"Le premier joueur gagne **{a['first_player_wins']}/100** parties. Longueur médiane : **{statistics.median(len(g['session']['history']) for g in games):g} tours** ; minimum {min(len(g['session']['history']) for g in games)}, maximum {max(len(g['session']['history']) for g in games)}.", '',
            'Causes finales des victoires :']
    lines += [f'- {reason} : {count}.' for reason,count in a['victory_reasons'].items()]
    lines += ['', 'Ces résultats décrivent cette version du moteur et ces placements. Ils ne constituent pas un classement universel des dieux : les groupes ne comptent que 8 ou 12 parties et leurs quatre variantes partagent un placement.', '',
              '## Mécanismes observés', '',
              '93 victoires viennent d’une montée au niveau 3, 5 de la condition de descente de Pan, 2 de l’absence de tour légal. Les victoires par blocage concernent une partie sans pouvoirs et une partie remportée par Atlas contre Pan.',
              'Dans Apollo contre Minotaur, 2 des 12 victoires d’Apollo se terminent directement par un échange avec un pion adverse sur la case de niveau 3 ; les dix autres se terminent par une montée ordinaire. Cela documente deux utilisations décisives du pouvoir, sans expliquer à lui seul le résultat de 12–0.', '',
              '## Fonctionnement de la recherche', '',
              f"Profondeur médiane : **{a['median_depth']:g}**. {a['stop_reasons'].get('timeout',0)} recherches se terminent faute de temps, {a['stop_reasons'].get('proof',0)} s’arrêtent sur un résultat annoncé comme démontré, {a['stop_reasons'].get('terminal',0)} constatent une position sans tour légal.",
              f"Durée moyenne du calcul : {s['average_search_seconds']:.2f} s ; médiane {a['median_search_seconds']:.2f} s. Le budget est une limite, pas une durée imposée lorsque la victoire ou la défaite est démontrée.", '',
              '| Pouvoir au trait | Profondeur médiane | Recherches |', '|---|---:|---:|']
    for power,v in a['depth_by_power'].items():
        lines.append(f"| {power} | {v['median']:g} | {v['turns']} |")
    lines += ['', 'Ces profondeurs dépendent aussi du pouvoir adverse et du stade de la partie ; elles incluent les arrêts anticipés après démonstration.', '',
              'Sur un placement identique au sol, le nombre de positions finales distinctes d’un tour est :']
    branching=json.loads((p/'branching-controlled.json').read_text())
    lines += [f"- {r['name']} : {r['unique_successors']} ({r['root_legal_turns']} séquences générées avant dédoublonnage)." for r in branching]
    lines += ['', 'Hermès possède donc ici environ 52 fois plus de successeurs distincts que les règles sans pouvoirs. Cette mesure explique un coût de recherche élevé ; elle ne mesure pas directement la force du pouvoir.', '',
              '## Exemple confirmé d’erreur évitable', '',
              'Partie 16, tour 26 : Héphaïstos joue contre Hermès. Avec 5 secondes, il choisit **B4 → C4**, puis construit deux blocs en **B4**, qui passe du niveau 1 au niveau 3. Son estimation est alors +430 à profondeur 7. Après ce tour, Hermès démontre une victoire forcée et gagne au tour 33.',
              'Une réanalyse à 20 secondes atteint la profondeur 8 et choisit le même déplacement, mais **une seule construction en D4**. L’estimation retombe à +62.',
              'Après cette alternative, une recherche du côté Hermès démontre au contraire sa défaite (-99992 à profondeur 8). La reprise de la partie avec 5 secondes par tour confirme **la victoire d’Héphaïstos au tour 34**.',
              'Le changement porte sur la décision de construction ; les réponses ultérieures sont recalculées. Dans la position originale, C4 est au sol et la double construction supprime l’accès direct à B4 en la portant au niveau 3. La construction en D4 crée un premier étage accessible. Cette géométrie est une explication plausible du contraste ; l’inversion du résultat est confirmée par la recherche et la reprise.', '',
              '## Pourquoi 20 secondes peuvent laisser la profondeur inchangée', '',
              'Partie 18, avant le tour 28 : la profondeur 5 est terminée en environ **1,2 seconde**. La profondeur 6 ne se termine qu’après **34,4 secondes**. Avec 5 ou 20 secondes, le résultat affiché reste donc celui de profondeur 5, même si le calcul continue.',
              'À 60 secondes, la profondeur 6 est achevée, la profondeur 7 demeure incomplète, le score passe de +414 à -154 et le coup proposé change. Cela illustre une estimation trop optimiste à un horizon limité, et non une limite volontaire à profondeur 3.', '',
              '## Priorités d’amélioration suggérées', '',
              '1. Ajouter des positions critiques de cette campagne à un corpus de comparaison, notamment les tours 26 de la partie 16 et 28 de la partie 18.',
              '2. Mieux ordonner les coups et approfondir sélectivement les positions tactiques : menaces de victoire, défenses forcées, constructions qui ferment une rampe. Respecter le budget global de 5 secondes et vérifier que les preuves restent valides.',
              '3. Tester une évaluation plus précise de l’accès effectif aux niveaux 2 et 3, des rampes et de l’immobilisation, ainsi que de la mobilité propre aux pouvoirs. Le cas Héphaïstos montre que la double construction facultative ne doit pas être favorisée automatiquement.',
              '4. Optimiser la génération et les équivalences de positions, particulièrement pour Hermès. Le moteur dédoublonne déjà les successeurs identiques ; les symétries des deux bâtisseurs pourraient offrir un gain supplémentaire si les actions sont remappées correctement.',
              '5. Confronter toute version modifiée à cette version figée, sur des matchs appariés à 5 secondes par tour et des placements supplémentaires. Une victoire contre soi-même ne démontre pas un progrès.', '',
              '## Audit et fichiers', '',
              '**Aucun historique invalide et aucune contradiction observée entre une victoire annoncée comme démontrée et le vainqueur final.** Cette vérification porte sur les règles implémentées et les parties observées ; les deux moteurs peuvent partager une erreur de règle.',
              '- `campaign.json` : protocole, graine, bibliothèque et 100 configurations.',
              '- `game-001.json` à `game-100.json` : parties complètes et décisions.',
              '- `summary.json`, `analysis.json` : résultats et audit.',
              '- `recheck-016-026.json`, `alternative-016-026.json`, `continuation-016-026.json` : erreur évitable et contrôle.',
              '- `recheck-018-028-60s.json` : chronologie des profondeurs.',
              '', 'Aucune stratégie de production n’a été modifiée ; cette campagne constitue une base mesurée pour les prochaines améliorations.']
    (p/'rapport.md').write_text('\n'.join(lines)+'\n')
    return p/'rapport.md'


if __name__=='__main__':
    print(report(sys.argv[1]))
