from __future__ import annotations
import os
import json
import numpy as np

DECK_MAGIC = b"DECKBIN1"
DECK_HEADER_SIZE = len(DECK_MAGIC) + 10


def saveDeck(deckList: list[str], filename:str, deckSize:int, chunkSize:int=1000000, overwrite: bool=False):
	'''Save decks as directory of files'''
	fileSplit = [a.tolist() for a in np.array_split(deckList,len(deckList)//chunkSize + 1)]
	file_path = f'data/{filename}_decks'
	os.makedirs(file_path, exist_ok=True)
	offset = max(1,len(os.listdir(file_path)))
	for d in range(len(fileSplit)):
		with open(f'{file_path}/{filename}_{d+offset}.bin', 'bw') as f:
			f.write(compress(fileSplit[d]))
	with open(f'{file_path}/metadata.json','w') as md:
		json.dump({'deckSize':deckSize,'chunkSize':chunkSize,'totalDecks':len(os.listdir(file_path))},md)

def compress(deckList: list[str]) -> bytearray:
	'''convert deck to binary file'''
	s = ''.join(deckList)
	i = 0
	buffer = bytearray()
	while i < len(s):
		buffer.append(int(s[i:i+8], 2))
		i += 8
	return buffer


def save_bin(deckList: list[str], filepath: str, deckSize: int = 52) -> None:
	"""save decks to one .bin file"""
	payload = compress(deckList)
	header = DECK_MAGIC + deckSize.to_bytes(2, "little") + len(deckList).to_bytes(8, "little")
	with open(filepath, "wb") as f:
		f.write(header)
		f.write(payload)


def load_bin(filepath: str, deckSize: int = 52) -> list[str]:
	"""load decks from one .bin file written by save_bin"""
	with open(filepath, "rb") as f:
		data = f.read()
	if data.startswith(DECK_MAGIC) and len(data) >= DECK_HEADER_SIZE:
		# decompress
		deckSize = int.from_bytes(data[len(DECK_MAGIC):len(DECK_MAGIC) + 2], "little")
		count = int.from_bytes(data[len(DECK_MAGIC) + 2:DECK_HEADER_SIZE], "little")
		payload = data[DECK_HEADER_SIZE:]
		bits = ''.join([format(w, '08b') for w in payload])
		bits = bits[:count * deckSize]
	else:
		bits = ''.join([format(w, '08b') for w in data])
		bits = bits[: (len(bits) // deckSize) * deckSize]
	return [''.join(item) for item in zip(*[iter(bits)] * deckSize)]


def append_bin(deckList: list[str], filepath: str, deckSize: int = 52) -> None:
	"""append decks to one .bin file written by save_bin"""
	# handles case where we want to create new bin file
	if not os.path.exists(filepath):
		save_bin(deckList, filepath, deckSize=deckSize)
		return
	with open(filepath, "rb") as f:
		data = f.read()
	if data.startswith(DECK_MAGIC) and len(data) >= DECK_HEADER_SIZE:
		# make sure we use little endian byte ordering
		current = int.from_bytes(data[len(DECK_MAGIC) + 2:DECK_HEADER_SIZE], "little")
		payload = compress(deckList)
		with open(filepath, "r+b") as f:
			# move to the end of the file
			f.seek(0, os.SEEK_END)
			f.write(payload)
			# adjust the post-magic accordingly
			f.seek(len(DECK_MAGIC) + 2)
			f.write((current + len(deckList)).to_bytes(8, "little"))
		return
	existing = load_bin(filepath, deckSize=deckSize)
	save_bin(existing + deckList, filepath, deckSize=deckSize)


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
