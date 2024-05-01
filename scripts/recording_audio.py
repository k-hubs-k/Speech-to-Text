import wave

import pyaudio


class AudioRecorder:
    def __init__(self, output_path):
        self.output_path = output_path

        # Paramètres d'enregistrement audio
        self.FORMAT = pyaudio.paInt16  # Format audio (16-bit PCM)
        self.CHANNELS = 1  # Nombre de canaux (mono)
        self.RATE = 44100  # Fréquence d'échantillonnage en Hz
        self.CHUNK = 1024  # Nombre d'échantillons par trame

    def record_audio(self):
        frames = []

        # Créer un objet PyAudio
        p = pyaudio.PyAudio()

        # Ouvrir un flux audio en entrée
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        print("Enregistrement audio en cours... (Appuyez sur Ctrl+C pour arrêter)")

        try:
            # Enregistrer le flux audio en temps réel
            while True:
                data = stream.read(self.CHUNK)
                frames.append(data)
        except KeyboardInterrupt:
            print("Enregistrement audio terminé.")

        # Fermer le flux audio
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Sauvegarder l'enregistrement audio dans un fichier WAV
        WAVE_OUTPUT_FILENAME = self.output_path + "/output.wav"
        wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b"".join(frames))
        wf.close()

    def record(self):
        print("Bienvenue dans l'enregistreur audio CLI.")
        print("Appuyez sur 'R' pour démarrer l'enregistrement.")
        print("Appuyez sur 'Q' pour quitter.")

        while True:
            user_input = input(">> ").strip().lower()

            if user_input == "r":
                self.record_audio()
            elif user_input == "q":
                print("Au revoir !")
                break
            else:
                print(
                    "Commande invalide. Veuillez entrer 'R' pour démarrer l'enregistrement ou 'Q' pour quitter."
                )


recorder = AudioRecorder("../res/recorded/")
recorder.record()
