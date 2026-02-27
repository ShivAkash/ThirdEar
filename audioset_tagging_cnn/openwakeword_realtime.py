import argparse
import time

import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
import noisereduce as nr


def run_realtime(args):
    openwakeword.utils.download_models()

    model = Model(wakeword_models=[args.model_path])

    sample_rate = 16000
    frame_samples = int(sample_rate * args.frame_ms / 1000)
    cooldown_s = args.cooldown_ms / 1000.0
    last_trigger = 0.0

    def audio_callback(indata, frames, time_info, status):
        nonlocal last_trigger
        if status:
            print(status)

        audio = (indata[:, 0] * 32768).astype(np.int16)
        
        # Apply Active Noise Cancelling before prediction
        if args.anc:
            # noisereduce requires floating point or integer arrays, but usually works best with float32 
            # or the native int16, we apply it directly to the 1D audio array
            audio = nr.reduce_noise(y=audio, sr=sample_rate)
            
        preds = model.predict(audio)

        score = 0.0
        if args.wakeword_name in preds:
            score = float(preds[args.wakeword_name])
        else:
            # Fallback to single-model output when label is unknown.
            score = float(list(preds.values())[0])

        now = time.time()
        if score >= args.threshold and (now - last_trigger) >= cooldown_s:
            last_trigger = now
            print(f"DETECTED: {args.display_name} (score={score:.2f})")

    with sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        blocksize=frame_samples,
        callback=audio_callback,
    ):
        print("Listening for wake word... Press Ctrl+C to stop.")
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time wake word detection using openWakeWord")
    parser.add_argument("--model_path", required=True, help="Path to the custom wake word model file")
    parser.add_argument("--wakeword_name", default="custom", help="Model label name, if known")
    parser.add_argument("--display_name", default="Shiv Akash", help="Name to print when detected")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--frame_ms", type=int, default=80, help="Audio frame size in milliseconds")
    parser.add_argument("--cooldown_ms", type=int, default=1000, help="Minimum ms between detections")
    parser.add_argument("--anc", action="store_true", help="Enable Active Noise Cancelling using noisereduce")
    args = parser.parse_args()

    run_realtime(args)
