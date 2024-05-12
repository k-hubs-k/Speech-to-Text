import numpy as np
from pydub.utils import json
import re
from sklearn.preprocessing import LabelEncoder

# Chargement des labels
X = np.load("mfcc_features.npy")

# Chargement des labels
with open("res/datasets/labels/audio_path.json", "r", encoding="utf-8") as f:
    transcriptions = json.load(f)

labels_train = np.array([transcription["transcription"] for transcription in transcriptions])

# Convertir les labels en un format numérique (par exemple, utiliser un encodage one-hot)
for i in range(len(labels_train)):
    # Conversion en minuscules
    labels_train[i] = labels_train[i].lower()
    # Suppression de la ponctuation
    labels_train[i] = re.sub(r'[^\w\s]', '', labels_train[i])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels_train)
print(y.shape)