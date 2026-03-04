from __future__ import annotations
import os
import json
import numpy as np
from src.decks import Deck

def save_decks(deck: Deck, filename: str, file_size: int = 0, overwrite: bool = False) -> None:
    """Save decks as directory of files of size `file_size`. A file size of `0` means keep everything in a single file"""
    deck_list = deck._decks
    deck_size = deck.deck_size
    if file_size > 0:
        chunk_size = file_size
    else:
        chunk_size = len(deck_list) * deck_size
    fileSplit = [a.tolist() for a in np.array_split(deck_list, len(deck_list) // chunk_size + 1)]
    file_path = f"data/{filename}"
    os.makedirs(file_path, exist_ok=True)
    offset = max(1, len(os.listdir(file_path)))
    for d in range(len(fileSplit)):
        with open(f"{file_path}/{filename}_{d+offset}.bin", "bw") as f:
            f.write(compress(fileSplit[d]))
    with open(f"{file_path}/metadata.json", "w") as md:
        json.dump(
            {
                "deck_size": deck_size,
                "chunk_size": chunk_size,
                "total_decks": len(deck_list),
                "total_deck_files": len(os.listdir(file_path)),
            },
            md,
        )


def compress(deckList: list[str]) -> bytearray:
    """
    Convert deck to binary file represented as hexadecimal
    
    Each card is represented as one bit, and each byte stores 8 cards. 
    """
    s = "".join(deckList)
    i = 0
    buffer = bytearray()
    while i < len(s):
        buffer.append(int(s[i : i + 8], 2))
        i += 8
    return buffer


def load_decks(foldername: str = "data/decktest_decks") -> Deck:
    """Decompress decks from directory of binary files."""
    deckList = []
    with open(f"{foldername}/metadata.json", "r") as mdj:  ## pull deck_size from metadata
        try:
            md = json.loads(mdj.read())
            deck_size = md["deck_size"]
        except KeyError:
            deck_size = 52

    for file in [file for file in os.listdir(foldername) if file.endswith(".bin")]:
        with open(f"{foldername}/{file}", "rb") as f:
            d = "".join([format(w, "08b") for w in f.read()])
        deckList += ["".join(item) for item in zip(*[iter(d)] * (deck_size))]
    return Deck(deckList)
