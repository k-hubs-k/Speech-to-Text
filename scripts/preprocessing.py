import json
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm


def normalize_audio_volume(audio_path, output_path, target_dBFS=-20.0):
    # Chargement de l'enregistrement audio
    audio = AudioSegment.from_file(audio_path)

    # Calcul du facteur de normalisation pour atteindre le niveau cible
    current_dBFS = audio.dBFS
    normalization_factor = target_dBFS - current_dBFS

    # Normalisation du volume de l'audio
    normalized_audio = audio + normalization_factor

    # Export de l'audio normalisé
    normalized_audio.export(output_path, format="wav")


def remove_background_noise(
    audio_path, output_path, noise_threshold=-40.0
):
    # Chargement de l'enregistrement audio
    audio = AudioSegment.from_file(audio_path)

    # Détection du bruit de fond
    background_noise = audio.dBFS

    # Filtrer le bruit de fond
    if background_noise > noise_threshold:
        audio = audio - noise_threshold
    else:
        audio = audio - background_noise

    # Export de l'audio filtré
    audio.export(output_path, format="wav")


def remove_silence(audio_path, output_file, silence_threshold=-45):
    # Charger le fichier audio
    audio = AudioSegment.from_file(audio_path, format="wav")

    # Détection des silences
    non_silent_audio = audio.strip_silence(silence_thresh=silence_threshold)

    # Exporter le fichier audio sans les silences
    non_silent_audio.export(output_file, format="wav")


def remove_artifacts(audio_path, output_path):
    # Chargement de l'enregistrement audio
    audio = AudioSegment.from_file(audio_path)

    # Suppression d'artefacts basée sur la fréquence ou l'amplitude
    # Par exemple, supprimer les fréquences inférieures à 1000 Hz
    audio_filtered = audio.low_pass_filter(1000)

    # Export de l'audio filtré
    audio_filtered.export(output_path, format="wav")


def temporal_normalization(audio_path, output_path, target_length):
    # Chargement de l'enregistrement audio
    audio = AudioSegment.from_file(audio_path)

    # Récupération de la durée actuelle de l'audio
    current_duration = int(len(audio))

    target_duration = target_length * 1000

    # print(f"target_duration : {target_duration}, current_duration : {current_duration}")

    # Calcul du facteur d'étirement/réduction nécessaire
    normalization_factor = target_duration / current_duration
    normalization_factor = 1.5

    # Application de la normalisation temporelle
    normalized_audio = audio.speedup(playback_speed=normalization_factor)

    # Export de l'audio normalisé
    normalized_audio.export(output_path, format="wav")


def convert_to_spectrogram(audio_path, output_path):
    # Chargement de l'enregistrement audio
    y, sr = librosa.load(audio_path)

    # Calcul du spectrogramme
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Affichage du spectrogramme (optionnel)
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    """plt.colorbar(format='%+2.0f dB')
    plt.title("spectogramme")
    plt.tight_layout()"""

    # Export du spectrogramme vers une image ou un fichier
    # plt.savefig(output_path)  # Sauvegarde de l'image
    # Ou exportez les données de spectrogramme dans un fichier
    np.save(output_path, spectrogram_db)

def extract_features_mfcc(audio_path, n_mfcc=13, desired_length=44100):
    # Chargement de l'enregistrement audio
    y, sr = librosa.load(audio_path, sr=None)

    # Normaliser la longueur de l'audio pour avoir une longueur fixe
    if len(y) < desired_length:
        # Remplir avec des zéros si le signal est trop court
        y = np.pad(y, (0, desired_length - len(y)), mode="constant")
    elif len(y) > desired_length:
        # Découper si le signal est trop long
        y = y[:desired_length]

    # Calcul des coefficients MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return mfccs


def Target_length(audio_dir):
    # Charger les fichiers audio et calculer leurs longueurs
    audio_files = []
    for filename in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, filename)
        audio_files.append(audio_path)

    audio_lengths = [librosa.get_duration(filename=file) for file in audio_files]

    # Déterminer la longueur cible (par exemple, la plus grande longueur)
    target_length = max(audio_lengths)
    return target_length

# Recuperation des labels
with open("res/datasets/labels/audio_path.json", "r", encoding="utf-8") as f:
    transcriptions = json.load(f)

labels = np.array([transcription["transcription"] for transcription in transcriptions])

# Chemin du dossier à parcourir
audio_dir = "res/datasets/audio_train/"
target_length = Target_length(audio_dir)
# Liste pour stocker les caractéristiques MFCC de chaque fichier audio
mfcc_features_list = []

# Parcourir tous les fichiers audio et extraire les coefficients MFCC
X_train = []

# Parcourir tous les fichiers audio et extraire les coefficients MFCC
for filename in tqdm(os.listdir(audio_dir)):
    if filename.endswith(".wav"):
        audio_path = os.path.join(audio_dir, filename)
        remove_silence(audio_path, "scripts/tmp/output_audio_without_silence.wav")
        normalize_audio_volume("scripts/tmp/output_audio_without_silence.wav","scripts/tmp/volume_normalized_audio.wav")
        remove_background_noise(
            "scripts/tmp/volume_normalized_audio.wav", "scripts/tmp/supp_bruit.wav"
        )
        remove_artifacts("scripts/tmp/supp_bruit.wav", "scripts/tmp/output_audio_filtered.wav")
        temporal_normalization(
            "scripts/tmp/output_audio_filtered.wav", "scripts/tmp/temporal_normalization.wav", 25
        )
        convert_to_spectrogram("scripts/tmp/temporal_normalization.wav", "scripts/tmp/spectrogram.npy")
        mfcc_features = extract_features_mfcc("scripts/tmp/temporal_normalization.wav")
        # Normalisation des caractéristiques MFCC (par exemple, en mettant à l'échelle entre -1 et 1)
        mfcc_features_normalized = (mfcc_features - np.mean(mfcc_features)) / np.std(
            mfcc_features
        )
        mfcc_features_list.append(mfcc_features_normalized)

# Convertir la liste en un tableau numpy
X = np.array(mfcc_features_list)
print(X.shape)
X = X.reshape(X.shape[0], -1)

# Enregistrement des caractéristiques prétraitées dans un fichier (optionnel)
np.save("mfcc_features.npy", X)