"""Initial resources and explicit parameters prescribed by the supplied booklet."""
from dataclasses import replace
from .extra import Extra, DIRECTIONS


def configure_extra(position, supplied=None):
    extra=supplied or Extra()
    reserves=list(extra.reserves)
    deck=list(extra.deck);discard=list(extra.discard)
    for player,power in enumerate(position.powers):
        if power==35:reserves[player]=2
        if power==31 and extra.wind not in DIRECTIONS:
            raise ValueError("Choisissez la direction initiale du vent d’Aeolus.")
        if power==42 and extra.siren[player] not in DIRECTIONS:
            raise ValueError("Choisissez la direction du chant de Siren.")
        if power==14:
            available=sum(1<<(n-1) for n in range(1,11) if n!=position.powers[1-player])
            card=extra.chaos[player]
            if not 1<=card<=10 or not available&(1<<(card-1)):
                raise ValueError("Indiquez la première carte de base tirée par Chaos, parmi les cartes inutilisées.")
            deck[player]=available&~(1<<(card-1));discard[player]=1<<(card-1)
        if power==43 and (extra.abyss[player] not in range(25) or position.occupied(extra.abyss[player])):
            raise ValueError("Choisissez une case initialement inoccupée pour l’abysse de Tartarus.")
        if power==40 and (extra.fate[player] not in range(20) or extra.fate[player]%5==4):
            raise ValueError("Choisissez le coin inférieur gauche d’une zone du Destin de 2 × 2 cases.")
    safe=list(extra.safe)
    for owner,power in enumerate(position.powers):
        if power==43:safe[owner]=sum(1<<c for ws in position.workers for c in ws if c>=0)
    return replace(extra,reserves=tuple(reserves),deck=tuple(deck),discard=tuple(discard),safe=tuple(safe))


def reference_setup(workers,powers,player,supplied=None):
    """Declared reference parameters for setup comparisons, not actual secrets."""
    from .engine import Position
    from .extra import initial_count
    workers=[list(ws) for ws in workers]
    occupied={c for ws in workers for c in ws}
    for owner,power in enumerate(powers):
        while len(workers[owner])<initial_count(power):
            cell=next(c for c in (12,2,22,10,14,*range(25)) if c not in occupied)
            workers[owner].append(cell);occupied.add(cell)
    free=next(c for c in range(25) if c not in occupied)
    extra=replace(Extra(),wind=6 if 31 in powers else -1,
        siren=tuple(13 if p==42 else -1 for p in powers),
        chaos=tuple(next(n for n in range(1,11) if n!=powers[1-i]) if p==14 else 0 for i,p in enumerate(powers)),
        abyss=tuple(free if p==43 else -1 for p in powers),
        fate=tuple(6 if p==40 else -1 for p in powers))
    if supplied:
        extra=replace(extra,wind=supplied.wind if supplied.wind>=0 else extra.wind,siren=tuple(a if a>=0 else b for a,b in zip(supplied.siren,extra.siren)),chaos=tuple(a or b for a,b in zip(supplied.chaos,extra.chaos)))
    pos=Position(workers=tuple(tuple(ws) for ws in workers),powers=tuple(powers),player=player,extra=extra)
    return replace(pos,extra=configure_extra(pos,extra))
