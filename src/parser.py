import numpy as np
import re
from typing import Literal, Tuple
from src.decks import Deck


class Parser:
	'''Parse deck list and get scores for each round'''
	
	playerOptionsType = Literal['000', '001', '010', '011', '100', '101', '110', '111'] # quick const for player option strings

	__slots__ = ('decks', '_playerOptions')
	def __init__(self, decks: Deck) -> None:
		self._playerOptions = set(['000', '001', '010', '011', '100', '101', '110', '111'])
		self.decks = decks
		return
	
	def winner(self, p1: playerOptionsType , p2: playerOptionsType):
		'''Find winner of each trick, return position and winner's choice'''
		def _matcher(deck, p1, p2):
			'''Handles the actual matching and scoring'''
			p1score = 0
			p2score = 0
			cardsLeft = deck
			while len(cardsLeft) != None:
				match = re.search(fr'({p1}|{p2})', cardsLeft)
				if match == None:
					break
				score = match.span()[1]
				if match.group() == p1:
					p1score += score
				elif match.group() == p2:
					p2score += score
				else:
					raise ValueError("Match found not equal to either player")
				cardsLeft = cardsLeft[score:]
			return (p1score, p2score)
		winners = [_matcher(w, p1,p2) for w in self.decks._decks]

		return np.unique_counts(np.argmax(winners,axis=1))
	

