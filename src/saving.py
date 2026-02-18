from __future__ import annotations
import numpy as np
import os
import json


def saveDeck(deckList: list[str], filename:str, deckSize:int, chunkSize:int=1000000):
	'''Save decks as directory of files'''
	fileSplit = [a.tolist() for a in np.array_split(deckList,len(deckList)//chunkSize + 1)]
	file_path = f'data/{filename}_decks'
	os.makedirs(file_path, exist_ok=True)
	offset = 1 + len(os.listdir(file_path))
	for d in range(len(fileSplit)):
		with open(f'{file_path}/{filename}_{d+offset}.bin', 'bw') as f:
			f.write(compress(fileSplit[d]))
	with open(f'{file_path}/metadata.json','w') as md:
		json.dump({'deckSize':deckSize,'chunkSize':chunkSize,'totalDecks':len(deckList)},md)

def compress(deckList: list[str]) -> bytearray:
	'''convert deck to binary file'''
	s = ''.join(deckList)
	i = 0
	buffer = bytearray()
	while i < len(s):
		buffer.append(int(s[i:i+8], 2))
		i += 8
	return buffer


def load(foldername:str='data/decktest_decks') -> list[str]:
	'''Decompress decks from directory of binary files.'''
	deckList = []
	with open(f'{foldername}/metadata.json','r') as mdj: ## pull decksize from metadata
		try:
			md = json.loads(mdj.read())
			deckSize = md['deckSize']
		except KeyError:
			deckSize = 52
	
	for file in [file for file in os.listdir(foldername) if file.endswith('.bin')]:
		with open(f'{foldername}/{file}','rb') as f:
			d = ''.join([format(w,'08b') for w in f.read()])
		deckList += [''.join(item) for item in zip(*[iter(d)]*(deckSize))]
	return deckList