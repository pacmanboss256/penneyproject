from __future__ import annotations
import numpy as np
import src.saving as saving


def deck_gen(
    num_decks: int = 1,
    deck_size: int = 52,
    save: bool = True,
    filename: str = "decktest",
    chunk_size: int = 10000,
    overwrite: bool = False,
) -> Deck:
    """
        Create n decks, and optionally save to a directory in chunks of n

        Deck Parameters:
        `num_decks` - total number of decks to generate\n
        `deck_size` - number of cards in each deck\n
        `save`      - option to save deck to a folder\n
        `filename`  - directory to save decks in\n
        `chunk_size`- how many decks per file\n
        `overwrite` - Replace decks with same name instead of append new ones
    """
    if deck_size % 2 == 1:
        raise ValueError("Deck size must be divisible by 2")
    baseDeck = np.concat(
        (np.zeros(deck_size // 2, dtype=int), np.ones(deck_size // 2, dtype=int))
    )  # get a deck of 1s and 0s
    allDeck = np.tile(baseDeck, (num_decks, 1))  # copy it a bunch
    deck_list = [
        "".join(x) for x in np.random.default_rng().permuted(allDeck, axis=1).astype(str).tolist()
    ]  # shuffle and convert to list of strings
    if save:
        saving.save_deck(deck_list, filename, deck_size=deck_size, chunk_size=chunk_size, overwrite=overwrite)
    x = Deck(deck_list)
    return x


def load_deck(foldername: str) -> Deck:
    """load saved files as deck object"""
    deck_list = saving.load(foldername)
    deck_obj = Deck(decks=deck_list)
    return deck_obj


class Deck:
    """Deck storage object, use `deck_gen()` or `load_deck()` to initialize decks"""

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

    def add_decks_from_file(self, filename: str):
        """Add decks to deck object from another file, assuming same deck size"""
        decks = load_deck(filename)
        self._decks = self._decks + decks._decks
        self._deck_count: int = len(self._decks)
