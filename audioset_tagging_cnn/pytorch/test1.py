import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model

openwakeword.utils.download_models()
model = Model()  # loads all bundled models

sample_rate = 16000
frame_ms = 80
frame_samples = int(sample_rate * frame_ms / 1000)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio = (indata[:, 0] * 32768).astype(np.int16)  # 16‑bit PCM
    preds = model.predict(audio)
    for k, v in preds.items():
        if v > 0.5:
            print(f"Detected: {k} ({v:.2f})")

with sd.InputStream(channels=1, samplerate=sample_rate, blocksize=frame_samples, callback=audio_callback):
    print("Listening... Ctrl+C to stop")
    while True:
        sd.sleep(1000)