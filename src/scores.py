import numpy as np
import pandas as pd
from numpy.typing import NDArray


type grid = list[tuple[str, str, NDArray]]

def createTable(data, score_by_tricks:bool) -> pd.DataFrame:
	'''Convert parser result to grid'''
	score_type = 'Cards'
	if score_by_tricks:
		score_type = 'Tricks'
	return pd.DataFrame(data,columns=['p1choice','p2choice',score_type]).pivot(index='p1choice',columns='p2choice',values=score_type)

class ScoreTable:
	def __init__(self, data: grid, scoring_by_tricks: bool=True):
		self.scoring = scoring_by_tricks
		self.table = createTable(data, scoring_by_tricks)

	def __repr__(self) -> str:
		return self.table.to_string()

  
	def addData(self, newData: grid) -> pd.DataFrame:
		'''Add a new parser result to existing score table object'''
		newTable = self.table + createTable(newData, self.scoring)
		self.table = newTable
		return newTable

	def saveData(self, filename: str) -> None:
		'''Save score table to csv file'''
		self.table.melt(ignore_index=False).reset_index().to_csv(f'scores/{filename}.csv', na_rep='--',index=False)
		return