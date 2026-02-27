import numpy as np
import pandas as pd
from numpy.typing import NDArray


type grid = list[tuple[str, str, NDArray]]

def createTable(data) -> pd.DataFrame:
	'''Convert parser result to grid'''
	return pd.DataFrame(data,columns=['p1choice','p2choice','result']).pivot(index='p1choice',columns='p2choice',values='result')

class ScoreTable:
	def __init__(self, data: grid):
		self.table = createTable(data)

	def __repr__(self) -> str:
		return self.table.to_string()

  
	def addData(self, newData: grid) -> pd.DataFrame:
		'''Add a new parser result to existing score table object'''
		newTable = self.table + createTable(newData)
		self.table = newTable
		return newTable

	def saveData(self, filename: str) -> None:
		'''Save score table to csv file'''
		self.table.melt(ignore_index=False).reset_index().to_csv(f'scores/{filename}.csv', na_rep='--',index=False)
		return