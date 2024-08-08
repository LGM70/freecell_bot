from typing import TypeAlias, Literal

class StandardSpec():
    """Standard freecell game specifications"""

    colors: list[str] = ['red', 'black']
    suits: list[str] = ['Diamonds', 'Clubs', 'Hearts', 'Spades']
    ranks: list[str] = ['A'] + [str(i) for i in range(2, 11)] + ['J', 'Q', 'K']
    loc_types: TypeAlias = Literal['cascade', 'cell', 'foundation']
    num_cards: int = len(suits) * len(ranks)
    num_cells: int = 4
    num_cascades: int = 8
    num_foundations: int = len(suits)
