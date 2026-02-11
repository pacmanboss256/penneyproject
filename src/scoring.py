import numpy
import re
from src.decks import Deck


class Parser:
	playerOptions = [format(x, '03b') for x in range(8)]
	def __init__(self, decks: Deck) -> None:
		self._decks = decks._decks
		self._deckCount = decks._deckCount
		self._deckSize = decks.size
		return
	
	