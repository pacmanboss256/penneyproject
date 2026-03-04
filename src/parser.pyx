import numpy as np
import re
from typing import Literal, Tuple
from src.decks import Deck, deck_gen

try:
    from src.fastmatch_simd import winner_counts_for_pair
except Exception:
    from src.fastmatch import winner_counts_for_pair
from itertools import permutations


class Parser:
    """Parse deck list and get scores for each round"""

    __slots__ = ("decks", "scores", "_decks_bytes", "bits", "scoring")

    def __init__(self, decks: Deck, bits: Literal[3, 4], scoring_by_tricks: bool = True) -> None:
        """Create a parser object for a Deck object

        params:
                decks: Deck object to parse
                bits: Whether to run the 3 or 4 bit game
                scoring_by_tricks: Whether or not to score by tricks won or total cards won

        """
        self.decks = decks
        self.scores = []
        self.bits = bits
        self._decks_bytes = [d.encode("ascii") for d in self.decks._decks]
        self.scoring = scoring_by_tricks
        return

    @property
    def player_options(self):
        return set([str(bin(w))[2:].zfill(self.bits) for w in range(2**self.bits)])

    @property
    def pairs(self):
        return list(permutations(self.player_options, 2))

    def winner2(self, p1, p2):  # returns a numpy unique counts result
        """Find winner of each trick, return position and winner's choice"""

        def _matcher_str(deck: str, p1, p2):
            """Handles the actual matching and scoring"""
            p1score = 0
            p2score = 0
            draw = 0
            newidx = 0
            cardsLeft = deck
            while len(cardsLeft) != None:
                p1match = cardsLeft.find(p1)
                p2match = cardsLeft.find(p2)
                if p2match == -1 and p1match == -1:  ## no match found
                    draw += 1
                    break
                elif (p2match < p1match) or (p1match == -1 and p2match > -1):
                    p2score += p2match + 3  ## index offset to include the cards in the match
                    newidx = p2match + 3
                elif (p1match < p2match) or (p2match == -1 and p1match > -1):
                    p1score += p1match + 3
                    newidx = p1match + 3
                cardsLeft = cardsLeft[newidx:]  # use rest of deck
            if p1score == p2score:
                draw += p2score + p1score + 1
            return [p1score, p2score, draw]

        winners = [_matcher_str(w, p1, p2) for w in self.decks._decks]
        outcomes = np.unique_counts(np.argmax(winners, axis=1)).counts
        return outcomes

    def winner(self, p1, p2) -> list:
        self.scores = list(
            winner_counts_for_pair(self._decks_bytes, p1, p2, aligned=False, score_by_tricks=self.scoring)
        )
        return self.scores

    def raw_out(self) -> list:
        """Output data as Tuple of str and numpy array"""
        res = []
        for i, j in self.pairs:
            w, l, t = self.winner(i, j)
            res.append([i, j, w, l, t])
        self.scores = res
        return res

    def add_decks(self, deck_count: int, decks=None) -> Parser:
        """
        generate deck_count additional decks (or use provided decks),
        score them, and update the parser
        """
        if decks is None:
            new_decks = deck_gen(num_decks=deck_count)
        else:
            if isinstance(decks, Deck):
                new_decks = decks
            elif isinstance(decks, list):
                new_decks = Deck(decks=decks)
            else:
                raise TypeError("decks must be a Deck or list[str]")

        new_bytes = [d.encode("ascii") for d in new_decks._decks]
        additional_scores = []
        for i, j in self.pairs:
            additional_scores.append(
                (i, j, winner_counts_for_pair(new_bytes, i, j, aligned=False, score_by_tricks=self.scoring))
            )
        for (indx, current_row), add_row in zip(enumerate(self.scores.copy()), additional_scores):
            add_w, add_l, add_t = add_row[2]
            self.scores[indx] = list(self.scores[indx])
            self.scores[indx][2] = int(current_row[2]) + int(add_w)
            self.scores[indx][3] = int(current_row[3]) + int(add_l)
            self.scores[indx][4] = int(current_row[4]) + int(add_t)
        self.decks._decks = self.decks._decks + new_decks._decks
        self.decks._deck_count = len(self.decks._decks)
        self._decks_bytes = self._decks_bytes + new_bytes
        return self.scores

    # backward compatibility
    def rawOut(self) -> list:
        return self.raw_out()
