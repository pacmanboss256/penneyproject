import src.decks
import src.parser
import src.saving
from src.heatmaps import make_heatmap
from src.parser import Parser
from src.decks import Deck, deckGen, loadDeck
import numpy as np
from textwrap import wrap
import pandas as pd

def main():
    print("Hello from penneyproject!")

    x = deckGen(numDecks=40000, chunkSize=10000, filename='decktest_deck4')
    #y = loadDeck('data/decktest_decks')
    w = Parser(x, bits=4)
    res = w.rawOut()
    print(pd.DataFrame.from_dict(res).sort_index().sort_index(axis=1))

    make_heatmap(res, parser=w)


if __name__ == "__main__":
    main()
