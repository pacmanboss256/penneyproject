from __future__ import annotations
import numpy as np
from datetime import datetime as dt
import src.saving as saving

def deckGen(numDecks: int=1, deckSize:int=52, save:bool=True, filename:str='decktest', chunkSize:int=1000000) -> Deck:
	'''Create n decks, and optionally save to a directory in chunks of n'''
	if deckSize % 2 == 1:
		raise ValueError("Deck size must be divisible by 2")
	baseDeck = np.concat((np.zeros(deckSize//2, dtype=int), np.ones(deckSize//2, dtype=int))) # type: ignore
	allDeck = np.tile(baseDeck,(numDecks, 1))
	deckList = ["".join(x) for x in np.random.default_rng().permuted(allDeck, axis=1).astype(str).tolist()]
	if save:
		saving.saveDeck(deckList, filename, deckSize=deckSize, chunkSize=chunkSize)
	x = Deck(deckList)
	return x

def loadDeck(foldername:str, decksize:int=52)-> Deck:
	'''load saved file as deck object'''
	deckList = saving.load(foldername)
	deckObj =  Deck(decks=deckList)
	return deckObj


class Deck:
	'''Class for decks of cards'''
	__slots__ = ('_decks','_deckCount','_deckSize')

	def __init__(self, decks):
		
		self._decks: list[str] = decks
		self._deckCount: int = len(self._decks)
		self._deckSize: int = len(self._decks[0])
		
	@property
	def deckSize(self):
		"""Number of cards in each deck"""
		return self._deckSize
		
	def __repr__(self) -> str:
		return '\n'.join(self._decks)
	
	def __getitem__(self, key:int) -> str:
		return self._decks[key]
	
	def __eq__(self, other) -> bool:
		if type(other) == Deck:
			return self._decks == other._decks
		else: return False

	def __bool__(self)-> bool:
		if self._deckCount == 0:
			return False
		else: return True

	def __len__(self) -> int:
		"""Number of decks contained in the object"""
		return self._deckCount
	
	def addDecksFromFile(self,filename:str):
		'''Add decks to deck object from another file, assuming same deck size'''
		decks = loadDeck(filename)
		self._decks = self._decks + decks._decks
		self._deckCount: int = len(self._decks)