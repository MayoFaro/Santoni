from dataclasses import dataclass


@dataclass(frozen=True)
class Power:
    number: int
    name: str
    family: str
    description: str
    supported: bool = True


_BASIC = [
    ("Apollo", "Peut échanger sa place avec un bâtisseur adverse."),
    ("Artemis", "Peut se déplacer deux fois sans revenir à sa case initiale."),
    ("Athena", "Après une montée, interdit les montées adverses au prochain tour."),
    ("Atlas", "Peut construire un dôme à toute hauteur."),
    ("Demeter", "Peut construire deux fois sur des cases différentes."),
    ("Hephaestus", "Peut ajouter un second bloc sur la même case, sans dôme."),
    ("Hermes", "Peut déplacer les deux bâtisseurs à hauteur constante, puis construire avec l'un."),
    ("Minotaur", "Peut pousser un adversaire sur la case libre située derrière lui."),
    ("Pan", "Gagne également en descendant d'au moins deux niveaux."),
    ("Prometheus", "Peut construire avant et après le déplacement, sans monter."),
]
_ADVANCED = [
    ("Aphrodite", "Un adversaire initialement adjacent doit terminer son déplacement adjacent.", True),
    ("Ares", "Peut retirer un bloc libre adjacent au bâtisseur resté immobile.", True),
    ("Bia", "Place ses bâtisseurs en premier ; élimine l'adversaire juste derrière sa destination.", True),
    ("Chaos", "Change de pouvoir par tirage après la construction d'un dôme.", True),
    ("Charon", "Peut forcer un adversaire adjacent de l'autre côté de son bâtisseur avant de bouger.", True),
    ("Chronus", "Gagne également dès qu'il existe cinq tours complètes.", True),
    ("Circe", "Peut s'approprier temporairement le pouvoir adverse.", True),
    ("Dionysus", "Une tour complète permet un tour supplémentaire avec un adversaire.", True),
    ("Eros", "Placement sur deux bords opposés ; gagne en rejoignant l'autre bâtisseur au niveau 1.", True),
    ("Hera", "L'adversaire ne gagne pas par un déplacement sur le bord.", True),
    ("Hestia", "Peut construire une deuxième fois hors du bord.", True),
    ("Hypnus", "Le bâtisseur adverse strictement plus haut que tous les autres ne bouge pas.", True),
    ("Limus", "Interdit les constructions adverses adjacentes, sauf un dôme de tour complète.", True),
    ("Medusa", "Construit sur un adversaire adjacent plus bas et l'élimine si possible.", True),
    ("Morpheus", "Accumule des matériaux pour construire zéro ou plusieurs fois.", True),
    ("Persephone", "L'adversaire doit monter au moins une fois si un tour légal le permet.", True),
    ("Poseidon", "Le bâtisseur immobile au sol peut construire jusqu'à trois fois.", True),
    ("Selene", "La bâtisseuse peut construire un dôme à toute hauteur à la place de la construction.", True),
    ("Triton", "Peut continuer à se déplacer après chaque arrivée sur le bord.", True),
    ("Zeus", "Peut construire un bloc sous lui, sans gagner par cette construction.", True),
    ("Aeolus", "Un jeton vent interdit une direction de déplacement.", True),
    ("Charybdis", "Place deux tourbillons reliant des cases.", True),
    ("Clio", "Les trois premières constructions portent des pièces bloquant l'adversaire.", True),
    ("Europa & Talus", "Déplace un jeton bloquant après son déplacement.", True),
    ("Gaea", "Peut ajouter des bâtisseurs après la construction d'un dôme.", True),
    ("Graeae", "Trois bâtisseurs ; peut choisir celui qui construit.", True),
    ("Hades", "Interdit les descentes adverses.", True),
    ("Harpies", "Prolonge les déplacements adverses dans la même direction.", True),
    ("Hecate", "Les bâtisseurs sont secrets.", True),
    ("Moerae", "Trois bâtisseurs et une zone secrète de victoire.", True),
    ("Nemesis", "Peut échanger les bâtisseurs des deux camps.", True),
    ("Siren", "Peut remplacer son tour par des déplacements forcés adverses.", True),
    ("Tartarus", "Une case secrète provoque une défaite immédiate.", True),
    ("Terpsichore", "Tous ses bâtisseurs doivent se déplacer puis construire.", True),
    ("Urania", "Les bords opposés sont adjacents pour ses déplacements et constructions.", True),
]
_HEROES = [
    ("Achilles", "Une fois : construire avant et après le déplacement."),
    ("Adonis", "Une fois : impose l'adjacence d'un bâtisseur adverse à la fin de son prochain tour, si possible."),
    ("Atalanta", "Une fois : enchaîner plusieurs déplacements."),
    ("Bellerophon", "Une fois : peut monter de deux niveaux."),
    ("Heracles", "Une fois : chaque bâtisseur peut construire des dômes sur ses cases adjacentes."),
    ("Jason", "Une fois : place un troisième bâtisseur au sol sur le bord ; celui-ci construit."),
    ("Medea", "Une fois : retire un bloc sous les bâtisseurs adjacents au bâtisseur immobile."),
    ("Odysseus", "Une fois : force des adversaires adjacents vers des coins libres."),
    ("Polyphemus", "Une fois : construit jusqu'à deux dômes sur n'importe quelles cases libres."),
    ("Theseus", "Une fois : élimine un adversaire adjacent exactement deux niveaux au-dessus."),
]

POWERS = {0: Power(0, "Aucun pouvoir", "none", "Règles ordinaires.")}
POWERS.update({i: Power(i, name, "basic", desc) for i, (name, desc) in enumerate(_BASIC, 1)})
POWERS.update({i: Power(i, name, "advanced", desc, ok) for i, (name, desc, ok) in enumerate(_ADVANCED, 11)})
POWERS.update({i: Power(i, name, "hero", desc) for i, (name, desc) in enumerate(_HEROES, 46)})

# Matches expressly marked N.R.C. in the supplied booklet (symmetric warning).
_NRC = {33: (17, 41), 35: (4, 41, 28), 36: (41,), 37: (9,),
        38: (7, 29), 39: (15, 17), 40: (39, 41),
        41: (11, 13, 24, 44, 55), 43: (13, 39, 40),
        44: (22, 23, 43), 45: (11,)}


def incompatible(a, b):
    return b in _NRC.get(a, ()) or a in _NRC.get(b, ())


# Perfect-information Arena excludes random draws and hidden objectives/workers.
ARENA_EXCLUDED = frozenset((14, 39, 40, 43))
