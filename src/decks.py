from __future__ import annotations
import numpy as np
from datetime import datetime as dt


def loadDeck(self, filename:str)-> Deck:
	with open(filename, 'r') as f:
		deckList = f.read().splitlines()
	deckSize = len(deckList[0])
	deckCount = len(deckList)
	deckObj =  Deck(deckSize=deckSize)
	deckObj._deckCount = deckCount
	deckObj._decks = deckList
	return deckObj


class Deck:
	'''Class to create decks of cards of size n'''
	__slots__ = ('size','_decks','_deckCount')

	def __init__(self, deckSize:int=52):
		if deckSize % 2:
			raise ValueError("Deck size must be divisible by 2")
		else:
			self.size = deckSize
		
		self._decks = list[str] | None
		self._deckCount: int = 0

		

	def deckGen(self, numDecks: int=1, saveDeck:bool=True, filename:str='') -> list[str]:
		baseDeck = np.concat((np.zeros(self.size//2, dtype=int), np.ones(self.size//2, dtype=int))) # type: ignore
		allDeck = np.tile(baseDeck,(numDecks, 1))
		deckList = ["".join(x) for x in np.random.default_rng().permuted(allDeck, axis=1).astype(str).tolist()]
		if saveDeck:
			if len(filename):
				file_path = f'data/{filename}'
			else:
				file_path = f'data/decks_{int(dt.timestamp(dt.now())*10)}.txt'
			with open(file_path, 'w+') as f:
				f.write('\n'.join(deckList))
		self._decks = deckList
		self._deckCount = numDecks
		return deckList
	



