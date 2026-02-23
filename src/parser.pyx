import numpy as np
import re
from typing import Literal, Tuple
from src.decks import Deck
from src.fastmatch import winner_counts_for_pair
from itertools import permutations

class Parser:
	'''Parse deck list and get scores for each round'''
	
	__slots__ = ('decks', 'scores', '_decks_bytes','bits')

	def __init__(self, decks: Deck, bits:Literal[3,4]) -> None:
		self.decks = decks
		self.scores = []
		self.bits = bits
		self._decks_bytes = [d.encode("ascii") for d in self.decks._decks]
		return

	@property
	def playerOptions(self):
		return set([str(bin(w))[2:].zfill(self.bits) for w in range(2**self.bits)])
		
	@property
	def PAIRS(self):
		return list(permutations(self.playerOptions,2))

	def winner2(self,p1,p2): # returns a numpy unique counts result 
		'''Find winner of each trick, return position and winner's choice'''

		def _matcher_str(deck:str, p1, p2):
			'''Handles the actual matching and scoring'''
			p1score = 0
			p2score = 0
			draw = 0
			newidx = 0
			cardsLeft = deck
			while len(cardsLeft) != None:
				p1match = cardsLeft.find(p1)
				p2match = cardsLeft.find(p2)
				if p2match == -1 and p1match == -1: ## no match found
					draw += 1
					break
				elif (p2match < p1match) or (p1match == -1 and p2match > -1): 
					p2score += p2match + 3 ## index offset to include the cards in the match
					newidx = p2match + 3
				elif (p1match < p2match) or (p2match == -1 and p1match > -1):
					p1score += p1match + 3
					newidx = p1match + 3
				cardsLeft = cardsLeft[newidx:] # use rest of deck
			if p1score == p2score:
				draw += p2score + p1score + 1
			return (p1score, p2score, draw)
		
		winners = [_matcher_str(w, p1,p2) for w in self.decks._decks]
		outcomes = np.unique_counts(np.argmax(winners,axis=1)).counts
		return outcomes	

	def winner(self, p1, p2):
		return winner_counts_for_pair(self._decks_bytes, p1, p2, aligned=False)

	def rawOut(self) -> list:
		'''Output data as Tuple of str and numpy array'''
		res = []
		for i,j in self.PAIRS:
			res.append((i,j,self.winner(i,j)))
		return res

	def allPairs(self) -> dict: 
		'''Output data as a neatly formatted list of lists'''
		res = {i: {j: (0,0,0) for j in self.playerOptions} for i in self.playerOptions}
		for i,j in self.PAIRS:
			res[i][j] = self.winner(i,j) # type: ignore
		return res