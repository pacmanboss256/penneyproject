from __future__ import annotations
import numpy as np
import src.saving as saving


def deck_gen(
    numDecks: int = 1,
    deck_size: int = 52,
    save: bool = True,
    filename: str = "decktest",
    chunkSize: int = 10000,
    overwrite: bool = False,
) -> Deck:
    """Create n decks, and optionally save to a directory in chunks of n"""
    if deck_size % 2 == 1:
        raise ValueError("Deck size must be divisible by 2")
    baseDeck = np.concat(
        (np.zeros(deck_size // 2, dtype=int), np.ones(deck_size // 2, dtype=int))
    )  # get a deck of 1s and 0s
    allDeck = np.tile(baseDeck, (numDecks, 1))  # copy it a bunch
    deckList = [
        "".join(x) for x in np.random.default_rng().permuted(allDeck, axis=1).astype(str).tolist()
    ]  # shuffle and convert to list of strings
    if save:
        saving.save_deck(deckList, filename, deck_size=deck_size, chunkSize=chunkSize, overwrite=overwrite)
    x = Deck(deckList)
    return x


def load_deck(foldername: str) -> Deck:
    """load saved file as deck object"""
    deckList = saving.load(foldername)
    deckObj = Deck(decks=deckList)
    return deckObj


class Deck:
    """Class for decks of cards"""

    __slots__ = ("_decks", "_deckCount", "_deck_size")

    def __init__(self, decks):

        self._decks: list[str] = decks
        self._deckCount: int = len(self._decks)
        self._deck_size: int = len(self._decks[0])

    @property
    def deck_size(self):
        """Number of cards in each deck"""
        return self._deck_size

    def __repr__(self) -> str:
        return "\n".join(self._decks)

    def __getitem__(self, key: int) -> str:
        return self._decks[key]

    def __eq__(self, other) -> bool:  ## deck equality defined by same deck content and dimensions
        if type(other) == Deck:
            return self._decks == other._decks
        else:
            return False

    def __bool__(self) -> bool:
        if self._deckCount == 0:
            return False
        else:
            return True

    def __len__(self) -> int:
        """Number of decks contained in the object"""
        return self._deckCount

    def add_decks_from_file(self, filename: str):
        """Add decks to deck object from another file, assuming same deck size"""
        decks = load_deck(filename)
        self._decks = self._decks + decks._decks
        self._deckCount: int = len(self._decks)
