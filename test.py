from importlib import reload
import src.decks
import src.saving
reload(src.saving);
reload(src.decks);
from src.parser import Parser
from src.heatmaps import make_heatmap
reload(src.heatmaps);
from src.decks import Deck, deck_gen
from src.saving import save_decks, load_decks
import numpy as np
from textwrap import wrap
import pandas as pd
from itertools import permutations

n = deck_gen(1000000,52)