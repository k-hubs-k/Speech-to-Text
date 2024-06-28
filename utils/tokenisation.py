import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

with open("./dataset/audio_path.json", "r", encoding="utf-8") as f:
    transcriptions = json.load(f)

labels = np.array([transcription["transcription"] for transcription in transcriptions])
data = labels

vocabularies = []
word_to_index = {}
index_to_word = {}
for sentence in data:
    tokens = sentence.split()
    for token in tokens:
        if token not in word_to_index:
            word_to_index[token] = len(word_to_index)
            index_to_word[len(index_to_word)] = token
            vocabularies.append(token)

# Encodage des séquences
encoded_data = []
for sentence in data:
    tokens = sentence.split()
    encoded_sentence = [word_to_index[token] for token in tokens]
    encoded_data.append(encoded_sentence)


# Création du Dataset et DataLoader
class LanguageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index])


dataset = LanguageDataset(encoded_data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

with open("./dataset/labels.json", "w") as f:
    for vocabulary in vocabularies:
        json.dump(vocabulary, f, ensure_ascii=False, indent=4)
