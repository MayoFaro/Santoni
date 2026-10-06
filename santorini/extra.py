"""Immutable, serializable state for the advanced cards and turn interruptions."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Extra:
    wind: int = -1
    siren: tuple[int, int] = (-1, -1)
    whirlpools: tuple[tuple[int, int], tuple[int, int]] = ((-1, -1), (-1, -1))
    coins: tuple[int, int] = (0, 0)
    coin_count: tuple[int, int] = (0, 0)
    talus: tuple[int, int] = (-1, -1)
    materials: tuple[int, int] = (0, 0)
    reserves: tuple[int, int] = (0, 0)
    chaos: tuple[int, int] = (0, 0)
    deck: tuple[int, int] = (0, 0)
    discard: tuple[int, int] = (0, 0)
    theft_owner: int = -1
    stolen: int = 0
    abyss: tuple[int, int] = (-1, -1)
    fate: tuple[int, int] = (-1, -1)
    event: int = 0
    event_owner: int = 0
    return_player: int = 0
    dome_cell: int = -1
    dion_owner: int = -1
    safe: tuple[int, int] = (0, 0)
    view: int = 0
    queued: int = 0
    queued_owner: int = 0
    queued_return: int = 0
    nemesis_active: int = 0
    original: tuple[int, int] = (255, 255)

    def validate(self):
        pairs=('siren','whirlpools','coins','coin_count','talus','materials','reserves','chaos','deck','discard','abyss','fate','safe','original')
        if any(not isinstance(getattr(self,key),tuple) or len(getattr(self,key))!=2 for key in pairs):
            raise ValueError('État de pouvoir : deux joueurs sont requis.')
        def ints(values,low,high):return all(type(x) is int and low<=x<=high for x in values)
        if self.wind not in (-1,*DIRECTIONS) or any(d not in (-1,*DIRECTIONS) for d in self.siren):
            raise ValueError('Direction de pouvoir invalide.')
        if any(not isinstance(pool,tuple) or len(pool)!=2 or not ints(pool,-1,24) or (pool[0]>=0 and pool[0]==pool[1]) for pool in self.whirlpools):
            raise ValueError('Tourbillons invalides.')
        if not ints(self.talus+self.abyss,-1,24) or any(c!=-1 and (not 0<=c<20 or c%5==4) for c in self.fate):
            raise ValueError('Jeton de pouvoir invalide.')
        if not ints(self.coins+self.safe,0,(1<<25)-1) or not ints(self.materials,0,(1<<32)-1):
            raise ValueError('Ressource de pouvoir invalide.')
        if not ints(self.coin_count,0,3) or not ints(self.reserves,0,2) or not ints(self.chaos,0,10) or not ints(self.deck+self.discard,0,1023):
            raise ValueError('Stock de pouvoir invalide.')
        if self.theft_owner not in (-1,0,1) or self.dion_owner not in (-1,0,1) or not 0<=self.stolen<=55:
            raise ValueError('Propriétaire de pouvoir invalide.')
        if self.event not in (0,1,2,3) or self.event_owner not in (0,1) or self.return_player not in (0,1) or self.dome_cell not in range(-1,25):
            raise ValueError('Intervention de pouvoir invalide.')
        if self.queued not in (0,3) or self.queued_owner not in (0,1) or self.queued_return not in (0,1) or self.view not in (0,1) or self.nemesis_active not in (0,1):
            raise ValueError('Contexte de pouvoir invalide.')
        if self.original!=(255,255) and not ints(self.original,0,55):
            raise ValueError('Pouvoirs d’origine invalides.')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        def freeze(value):
            return tuple(freeze(x) for x in value) if isinstance(value, (tuple, list)) else value
        return cls(**{key: freeze(value) for key, value in data.items()})


NEW_POWERS = frozenset((14, 17, 18, 25, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44))
SECRET_POWERS = frozenset((39, 40, 43))
DIRECTIONS = (6, 7, 8, 11, 13, 16, 17, 18)


def requires_native(position):
    return bool(NEW_POWERS.intersection(position.powers) or position.extra != Extra() or position.resume)


def initial_count(power):
    return 3 if power in (36, 40) else 2
