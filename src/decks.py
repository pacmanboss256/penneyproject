from __future__ import annotations
import numpy as np
from datetime import datetime as dt


def loadDeck(filename:str)-> Deck:
	'''load saved file as deck object'''
	with open(filename, 'r') as f:
		deckList = f.read().splitlines()
	deckObj =  Deck(decks=deckList)
	return deckObj

def deckGen(numDecks: int=1, deckSize:int=52, saveDeck:bool=True, filename:str='decktest.txt') -> Deck:
	'''Create n decks, and optionally save to a file'''
	if deckSize % 2 == 1:
		raise ValueError("Deck size must be divisible by 2")
	baseDeck = np.concat((np.zeros(deckSize//2, dtype=int), np.ones(deckSize//2, dtype=int))) # type: ignore
	allDeck = np.tile(baseDeck,(numDecks, 1))
	deckList = ["".join(x) for x in np.random.default_rng().permuted(allDeck, axis=1).astype(str).tolist()]
	if saveDeck:
		if len(filename):
			file_path = f'data/{filename}'
		else:
			file_path = f'data/decks_{int(dt.timestamp(dt.now())*10)}.txt'
		with open(file_path, 'w+') as f:
			f.write('\n'.join(deckList))
	x = Deck(deckList)
	return x
	
class Deck:
	'''Class for deck of cards'''
	__slots__ = ('_decks','_deckCount','_deckSize')

	def __init__(self, decks):
		
		self._decks: list[str] = decks
		self._deckCount: int = len(self._decks)
		self._deckSize: int = len(self._decks[0])

	def __repr__(self) -> str:
		return '\n'.join(self._decks)
	
	def __getitem__(self, key:int) -> str:
		return self._decks[key]





