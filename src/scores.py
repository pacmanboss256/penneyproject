from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from itertools import permutations


def load_table(filename:str) -> ScoreTable:
    w = pd.read_csv(f'scores/{filename}',dtype={'p1choice':str,'p2choice':str}).values.tolist()
    return ScoreTable(w)

class ScoreTable:
    def __init__(self, data, scoring_by_tricks: bool = True):
        self.scoring = scoring_by_tricks
        self.raw = pd.DataFrame(data, columns=['p1choice','p2choice','win','loss','tie'])
        self.table = self._create_table()

    def _create_table(self) -> pd.DataFrame:
        return pd.concat([self.raw[['p1choice','p2choice']],pd.Series(self.raw[['win','loss','tie']].values.tolist(),name='wlt')],axis=1).pivot(columns='p1choice',index='p2choice',values='wlt')

    def save(self, filename: str) -> None:
        '''Save score table as csv file'''
        self.raw.to_csv(f'scores/{filename}', na_rep='NaN', index=False)


    def __eq__(self, other) -> bool:
        if type(other) == ScoreTable:
            return pd.DataFrame.equals(self.raw, other.raw)
        else: return False

    def __repr__(self) -> str:
        return self.table.to_string()