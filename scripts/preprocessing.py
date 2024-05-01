import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def normalize_audio_volume(audio_path, target_dBFS=-20.0):
    # Chargement de l'enregistrement audio
    audio = AudioSegment.from_file(audio_path)

    # Calcul du facteur de normalisation pour atteindre le niveau cible
    current_dBFS = audio.dBFS
    normalization_factor = target_dBFS - current_dBFS

    # Normalisation du volume de l'audio
    normalized_audio = audio + normalization_factor

    # Export de l'audio normalisé
    normalized_audio.export("./tmp/volume_normalized_audio.wav", format="wav")


def remove_background_noise(
    audio_path, output_path="./tmp/supp_bruit.wav", noise_threshold=-40.0
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


def speech_segmentation(audio_path, min_silence_len=500, silence_thresh=-30):
    # Chargement de l'enregistrement audio
    audio = AudioSegment.from_file(audio_path)

    # Segmentation de la parole basée sur le silence
    segments = split_on_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )

    # Export des segments de parole
    for i, segment in enumerate(segments):
        segment.export(f"./tmp/segment_{i+1}.mp4", format="mp4")


def remove_artifacts(audio_path, output_path):
    # Chargement de l'enregistrement audio
    audio = AudioSegment.from_file(audio_path)

    # Suppression d'artefacts basée sur la fréquence ou l'amplitude
    # Par exemple, supprimer les fréquences inférieures à 1000 Hz
    audio_filtered = audio.low_pass_filter(1000)

    # Export de l'audio filtré
    audio_filtered.export(output_path, format="wav")


def temporal_normalization(audio_path, filename, target_duration=12000):
    # Chargement de l'enregistrement audio
    audio = AudioSegment.from_file(audio_path)

    # Récupération de la durée actuelle de l'audio
    current_duration = len(audio)
    # print(current_duration)

    # Calcul du facteur d'étirement/réduction nécessaire
    normalization_factor = target_duration / current_duration

    # Application de la normalisation temporelle
    normalized_audio = audio.speedup(playback_speed=normalization_factor)

    # Export de l'audio normalisé
    normalized_audio.export(filename, format="wav")


def convert_to_spectrogram(audio_path, output_path):
    # Chargement de l'enregistrement audio
    y, sr = librosa.load(audio_path)

    # Calcul du spectrogramme
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Affichage du spectrogramme (optionnel)
    """librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("spectogramme")
    plt.tight_layout()"""

    # Export du spectrogramme vers une image ou un fichier
    plt.savefig(output_path)  # Sauvegarde de l'image
    # Ou exportez les données de spectrogramme dans un fichier
    np.save(output_path, spectrogram)


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


# Recuperation des labels
with open("../res/datasets/labels/audio_path.json", "r", encoding="utf-8") as f:
    transcriptions = json.load(f)

labels = np.array([transcription["transcription"] for transcription in transcriptions])

# Chemin du dossier à parcourir
audio_dir = "../res/datasets/audio_train/"

# Parcourir tous les fichiers audio et extraire les coefficients MFCC
for filename in tqdm(os.listdir(audio_dir)):
    if filename.endswith(".wav"):
        audio_path = os.path.join(audio_dir, filename)
        normalize_audio_volume(audio_path)
        remove_background_noise(
            "./tmp/volume_normalized_audio.wav", "./tmp/supp_bruit.wav"
        )
        speech_segmentation("./tmp/supp_bruit.wav")
        remove_artifacts("./tmp/segment_1.mp4", "./tmp/output_audio_filtered.wav")
        temporal_normalization(
            "./tmp/output_audio_filtered.wav", "../res/analysed_data/" + filename
        )
        output_path = "./tmp/spectrogram.png"
        convert_to_spectrogram("./tmp/normalized_audio.wav", output_path)
        mfcc_features = extract_features_mfcc("../res/analysed_data/" + filename)
        # Normalisation des caractéristiques MFCC (par exemple, en mettant à l'échelle entre -1 et 1)
        mfcc_features_normalized = (mfcc_features - np.mean(mfcc_features)) / np.std(
            mfcc_features
        )

        # Pour stocker les caractéristiques MFCC de chaque fichier audio
        np.save("../res/mfcc_data/" + filename + ".npy", mfcc_features_normalized)
