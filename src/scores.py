from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _resolve_score_path(filename: str | Path) -> Path:
    """
    Resolve a score path
    """
    path = Path(filename)
    if not path.is_absolute() and path.parent == Path("."):
        path = Path("scores") / path
    return path


def load_table(filename: str | Path, scoring_by_tricks: bool = True) -> ScoreTable:
    """Load scores from a .csv file."""
    path = _resolve_score_path(filename)
    w = pd.read_csv(path, dtype={"p1choice": str, "p2choice": str}).values.tolist()
    return ScoreTable(w, scoring_by_tricks)


class ScoreTable:
    def __init__(self, data, scoring_by_tricks: bool = True):
        """Takes a list returned by `Parser.rawOut()`"""
        self.scoring = scoring_by_tricks
        self.raw = pd.DataFrame(data, columns=["p1choice", "p2choice", "win", "loss", "tie"])
        self.table = self._create_table()

    def _create_table(self) -> pd.DataFrame:
        """Creates a heatmap-like format from raw data"""
        return pd.concat(
            [
                self.raw[["p1choice", "p2choice"]],
                pd.Series(self.raw[["win", "loss", "tie"]].values.tolist(), name="wlt"),
            ],
            axis=1,
        ).pivot(columns="p1choice", index="p2choice", values="wlt")

       
    def addData(self, other: ScoreTable) -> None:
        """Add data from other scoretable object"""
        self.raw[['win','loss','tie']] += other.raw[['win','loss','tie']]
        self.table = self._create_table()
        return

    def save(self, filename: str) -> None:
        """Save score table as csv file."""
        path = _resolve_score_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw.to_csv(path, na_rep="NaN", index=False)
        return

    def __eq__(self, other) -> bool: ## scoretables are equal if the scores and choices are identical
        if type(other) == ScoreTable:
            return pd.DataFrame.equals(self.raw, other.raw)
        else:
            return False

    def __repr__(self) -> str:
        return self.table.to_string()
