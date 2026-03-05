from __future__ import annotations
import numpy as np


def deck_gen(
    num_decks: int = 1,
    deck_size: int = 52,
) -> Deck:
    """
        Deck Parameters:
        `num_decks` - total number of decks to generate\n
        `deck_size` - number of cards in each deck\n
    """
    if deck_size % 2 == 1:
        raise ValueError("Deck size must be divisible by 2")
    base_deck = np.concat(
        (np.zeros(deck_size // 2, dtype=int), np.ones(deck_size // 2, dtype=int))
    )  # get a deck of 1s and 0s
    all_deck = np.tile(base_deck, (num_decks, 1))  # copy it a bunch
    deck_list = [
        "".join(x) for x in np.random.default_rng().permuted(all_deck, axis=1).astype(str).tolist()
    ]  # shuffle and convert to list of strings
    x = Deck(deck_list)
    return x


class Deck:
    """Deck object. Use `deck_gen()` or `saving.load()` to create decks"""

    __slots__ = ("_decks", "_deck_count", "_deck_size")

    def __init__(self, decks):

        self._decks: list[str] = decks
        self._deck_count: int = len(self._decks)
        self._deck_size: int = len(self._decks[0])

    @property
    def deck_size(self):
        """Number of cards in each deck"""
        return self._deck_size
    
    @property
    def decks(self):
        """Raw list of decks"""
        return self._decks
    

    def __repr__(self) -> str:
        return "\n".join(self._decks)

    def __getitem__(self, key: int) -> str: # index directly into deck object 
        return self._decks[key]

    def __eq__(self, other) -> bool:  ## deck equality defined by same deck content and dimensions
        if type(other) == Deck:
            return self._decks == other._decks
        else:
            return False

    def __bool__(self) -> bool: # truthy value determined by having at least one deck and a non-zero length
        if self._deck_count == 0 or self.deck_size == 0:
            return False
        else:
            return True

    def __len__(self) -> int:
        """Number of decks contained in the object"""
        return self._deck_count

    def add_decks(self, other: Deck):
        if other.deck_size != self.deck_size:
            Warning("Deck sizes do not match")
            return
        else:
            self._decks += other._decks
            self._deck_count = len(self._decks)
            print(f"{other._deck_count} decks successfully added")
            return
