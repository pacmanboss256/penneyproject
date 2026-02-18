import numpy as np
import pandas as pd


class ScoreTable:
	def __init__(self, data: dict):
		self.table = pd.DataFrame.from_dict(data).sort_index().sort_index(axis=1)

	def addData(self,data:dict):
		self.table += pd.DataFrame.from_dict(data).sort_index().sort_index(axis=1)