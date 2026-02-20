import numpy as np
import re
from typing import Literal, Tuple
from src.decks import Deck
from src.fastmatch import winner_counts_for_pair

class Parser:
	'''Parse deck list and get scores for each round'''
	
	playerOptionsType = Literal['000', '001', '010', '011', '100', '101', '110', '111'] # quick const for player option strings

	## hardcoded pairs for looping winner
	PAIRS: list[Tuple[playerOptionsType, playerOptionsType]] = [('000','001'),('000','010'),('000','011'),('000','100'),('000','101'),('000','110'),('000','111'),('001','000'),
		     ('001','010'),('001','011'),('001','100'),('001','101'),('001','110'),('001','111'),('010','000'),('010','001'),
		     ('010','011'),('010','100'),('010','101'),('010','110'),('010','111'),('011','000'),('011','001'),('011','010'),
		     ('011','100'),('011','101'),('011','110'),('011','111'),('100','000'),('100','001'),('100','010'),('100','011'),
		     ('100','101'),('100','110'),('100','111'),('101','000'),('101','001'),('101','010'),('101','011'),('101','100'),
		     ('101','110'),('101','111'),('110','000'),('110','001'),('110','010'),('110','011'),('110','100'),('110','101'),
		     ('110','111'),('111','000'),('111','001'),('111','010'),('111','011'),('111','100'),('111','101'),('111','110')]
	
	playerOptions = set(['000', '001', '010', '011', '100', '101', '110', '111'])

	__slots__ = ('decks', 'scores', '_decks_bytes')

	def __init__(self, decks: Deck) -> None:
		self.decks = decks
		self.scores = []
		self._decks_bytes = [d.encode("ascii") for d in self.decks._decks]
		return
	
	def winner2(self,p1: playerOptionsType , p2: playerOptionsType): # returns a numpy unique counts result 
		'''Find winner of each trick, return position and winner's choice'''

		def _matcher_str(deck:str, p1, p2):
			'''Handles the actual matching and scoring'''
			p1score = 0
			p2score = 0
			newidx = 0
			cardsLeft = deck
			while len(cardsLeft) != None:
				p1match = cardsLeft.find(p1)
				p2match = cardsLeft.find(p2)
				if p2match == -1 and p1match == -1: ## no match found
					break
				elif (p2match < p1match) or (p1match == -1 and p2match > -1): 
					p2score += p2match + 3 ## index offset to include the cards in the match
					newidx = p2match + 3
				elif (p1match < p2match) or (p2match == -1 and p1match > -1):
					p1score += p1match + 3
					newidx = p1match + 3
				cardsLeft = cardsLeft[newidx:] # use rest of deck
			if p1score == p2score: draw += 1
			return (p1score, p2score, draw)
		
		winners = [_matcher_str(w, p1,p2) for w in self.decks._decks]
		outcomes = np.unique_counts(np.argmax(winners,axis=1)).counts
		return outcomes
	
	def rawOut(self) -> list:
		'''Output data as Tuple of str and numpy array'''
		res = []
		for i,j in self.PAIRS:
			res.append((i,j,self.winner(i,j)))
		return res

	def allPairs(self) -> dict: 
		'''Output data as a neatly formatted list of lists'''
		res = {i: {j: (0,0,0) for j in Parser.playerOptions} for i in Parser.playerOptions}
		for i,j in self.PAIRS:
			res[i][j] = self.winner(i,j) # type: ignore
		return res
