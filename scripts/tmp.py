import ipywidgets as widgets
from IPython.display import display, Audio
import sounddevice as sd
import numpy as np
import threading

class AudioRecorder:
    def __init__(self):
        self.audio = widgets.Output()
        self.record_button = widgets.Button(description='Enregistrer', disabled=False, button_style='info')
        self.stop_button = widgets.Button(description='Arrêter', disabled=True, button_style='danger')
        self.record_button.on_click(self.start_recording)
        self.stop_button.on_click(self.stop_recording)
        self.ui = widgets.VBox([self.record_button, self.stop_button, self.audio])
        self.frames = []
        self.is_recording = False

    def start_recording(self, _):
        self.frames = []
        self.is_recording = True
        self.record_button.disabled = True
        self.stop_button.disabled = False
        self.audio.clear_output(wait=True)

        def record():
            while self.is_recording:
                data = sd.rec(1024, samplerate=44100, channels=1, dtype=np.int16)
                self.frames.extend(data)
                sd.wait()

        threading.Thread(target=record).start()

    def stop_recording(self, _):
        self.is_recording = False
        self.record_button.disabled = False
        self.stop_button.disabled = True
        self.audio.clear_output(wait=True)
        with self.audio:
            display(Audio(np.vstack(self.frames), rate=44100))

recorder = AudioRecorder()
display(recorder.ui)
recorder.start_recording("hubs")