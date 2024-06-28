import json
import os

import librosa
import numpy as np
import soundfile as sf
import torchaudio
from pydub import AudioSegment
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from torchaudio.transforms import MelSpectrogram
from torchvision.transforms import Compose


def load_mean_length(file_src="./dataset/mean_length.json"):
    if not os.path.exists(file_src):
        print(
            "Le fichier contenant la longueur moyenne n'existe pas. Cela peut affecter la pprecision du modele"
        )
        return 0.4374343333333333

    with open(file_src, "r", encoding="utf-8") as f:
        return json.load(f)


class Custom_preprocessing:
    def __init__(self, audio_path, output_file):
        self.audio = audio_path
        # Charger les fichiers audio et calculer leurs longueurs
        self.output_file = output_file

        self.target_length = load_mean_length()
        self.process()

    # Élimination des silences
    def remove_silence(self, audio_path, output):
        # Charger le fichier audio
        signal, sr = librosa.load(audio_path, sr=None)

        # Détection des régions actives dans le signal
        non_silent_intervals = librosa.effects.split(signal, top_db=30)

        # Fusionner les intervalles non silencieux
        non_silent_signal = librosa.effects.remix(signal, non_silent_intervals)

        # Sauvegarder le signal audio sans les silences
        sf.write(output, non_silent_signal, sr)

    # Normalisation du volume
    def normalize_audio_volume(self, audio_path, output_path, target_dBFS=-20.0):
        # Chargement de l'enregistrement audio
        audio = AudioSegment.from_file(audio_path)

        # Calcul du facteur de normalisation pour atteindre le niveau cible
        current_dBFS = audio.dBFS
        normalization_factor = target_dBFS - current_dBFS

        # Normalisation du volume de l'audio
        normalized_audio = audio + normalization_factor

        # Export de l'audio normalisé
        normalized_audio.export(output_path, format="wav")

    # Filtrage du bruit
    def filtrage_du_bruit(self, audio_path, output, noise_threshold=-40.0):
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
        audio.export(output, format="wav")

    # Segmentation de la parole
    def segmentation_parole(self, audio_path, output_file, silence_threshold=-45):
        # Charger le fichier audio
        audio = AudioSegment.from_file(audio_path, format="wav")

        # Détection des silences
        non_silent_audio = audio.strip_silence(silence_thresh=silence_threshold)

        # Exporter le fichier audio sans les silences
        non_silent_audio.export(output_file, format="wav")

    # Éliminé des artefacts
    def remove_artifacts(self, audio_path, output_path):
        # Chargement de l'enregistrement audio
        audio = AudioSegment.from_file(audio_path)

        # Suppression d'artefacts basée sur la fréquence ou l'amplitude
        # Par exemple, supprimer les fréquences inférieures à 1000 Hz
        audio_filtered = audio.low_pass_filter(1000)

        # Export de l'audio filtré
        audio_filtered.export(output_path, format="wav")

    # Préaccentuation
    def preaccentuation(self, audio_file, output):
        # Charger le signal vocal (remplacer "audio.wav" par votre propre fichier audio)
        sample_rate, audio_data = wavfile.read(audio_file)

        # Paramètres de la préaccentuation
        alpha = 0.95  # Facteur de préaccentuation (typiquement entre 0.9 et 1)

        # Appliquer la préaccentuation
        preemphasis_audio = np.append(
            audio_data[0], audio_data[1:] - alpha * audio_data[:-1]
        )

        # Enregistrer le signal filtré en tant que fichier WAV
        wavfile.write(output, sample_rate, np.int16(preemphasis_audio))

    # Normalisation temporelle
    def time_stretch_audio(self, input_file, output_file, target_duration):
        # Charger l'audio
        audio, sr = librosa.load(input_file)

        # Calculer la durée actuelle
        current_duration = len(audio) / 44100

        # Calculer le facteur de normalisation
        speed_factor = current_duration / target_duration

        # Normaliser l'audio en modifiant la vitesse
        normalized_audio = librosa.effects.time_stretch(y=audio, rate=speed_factor)

        # Sauvegarder l'audio normalisé
        sf.write(output_file, normalized_audio, sr)

    # Convertir en_format adapté
    def convert_to_spectrogram(self, audio_path):
        transform = Compose(
            [
                torchaudio.transforms.Resample(
                    orig_freq=44100, new_freq=16000
                ),  # Échantillonnage à 16 kHz
                MelSpectrogram(
                    n_fft=400, win_length=400, hop_length=160, n_mels=128
                ),  # Créer un spectrogramme Mel
            ]
        )
        waveform, sample_rate = torchaudio.load(audio_path)
        spectrogram = transform(waveform).unsqueeze(0)  # Ajouter une dimension de lot
        spectrogram = np.array(spectrogram)
        spectrogram = spectrogram.reshape(-1)

        return spectrogram, sample_rate

    # Correction d'accent
    def correction_accent(self, audio_file):
        signal, sample_rate = librosa.load(audio_file, sr=None)

        # Calculer les coefficients cepstraux MFCC
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)

        # Normaliser les coefficients MFCC
        scaler = StandardScaler()
        return scaler.fit_transform(mfccs.T).T

    def process(self):
        mfcc_features_list = []
        self.remove_silence(self.audio, "remove_silence.wav")
        self.normalize_audio_volume("remove_silence.wav", "normalize_audio_volume.wav")
        self.time_stretch_audio(
            "normalize_audio_volume.wav", "time_stretch_audio.wav", self.target_length
        )
        self.filtrage_du_bruit("time_stretch_audio.wav", "filtrage_du_bruit.wav")
        self.segmentation_parole("filtrage_du_bruit.wav", "segmentation_parole.wav")
        self.remove_artifacts("segmentation_parole.wav", "remove_artifacts.wav")
        self.preaccentuation("remove_artifacts.wav", "preaccentuation.wav")
        mfcc_features_normalized = self.correction_accent("preaccentuation.wav")
        mfcc_features_list.append(mfcc_features_normalized)

        X = np.array(mfcc_features_list)
        X = X.reshape(X.shape[0], -1)

        # Enregistrement des caractéristiques prétraitées dans un fichier (optionnel)
        np.save(self.output_file, X)


audio_ = "200704-112542_plt_cfe_elicit_5.wav"
test = Custom_preprocessing(audio_, "works.npy")
