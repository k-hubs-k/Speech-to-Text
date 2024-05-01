import os

import numpy as np
from pydub.utils import json

audio_dir = "../res/mfcc_data/"

mfcc_lists = []

# Chargement des caracteristiques
for filename in os.listdir(audio_dir):
    if filename.endswith(".npy"):
        file_path = os.path.join(audio_dir, filename)

        # Chargement du fichier current
        mfcc = np.load(file_path)
        mfcc_lists.append(mfcc)


X = np.array(mfcc_lists)

# Chargement des labels
with open("../res/datasets/labels/audio_path.json", "r", encoding="utf-8") as f:
    transcriptions = json.load(f)

y = np.array([transcription["transcription"] for transcription in transcriptions])
