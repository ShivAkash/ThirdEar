import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import torch
import sounddevice as sd
from collections import deque
import threading

from models import *
from pytorch_utils import move_data_to_device
import config


def realtime_audio_tagging(args):
    """Real-time audio tagging from microphone."""
    
    # Arguments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    chunk_duration = args.chunk_duration  # seconds
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    noise_reducer = None
    if args.anc:
        try:
            import noisereduce as nr
            noise_reducer = nr
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "ANC is enabled (--anc), but 'noisereduce' is not installed. "
                "Install it with: pip install noisereduce"
            )

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    print(f'Recording from microphone with sample rate {sample_rate} Hz...')
    print('Press Ctrl+C to stop.')
    
    chunk_samples = int(chunk_duration * sample_rate)
    audio_buffer = deque(maxlen=chunk_samples)
    
    def audio_callback(indata, frames, time, status):
        if status:
            print(f'Audio error: {status}')
        audio_buffer.extend(indata[:, 0])
    
    # Start recording
    with sd.InputStream(samplerate=sample_rate, channels=1, 
                       blocksize=chunk_samples, callback=audio_callback):
        try:
            while True:
                if len(audio_buffer) == chunk_samples:
                    # Convert to numpy array
                    audio_chunk = np.array(list(audio_buffer))
                    audio_buffer.clear()
                    
                    if args.anc:
                        audio_chunk = noise_reducer.reduce_noise(y=audio_chunk, sr=sample_rate)
                        
                    if args.play_audio:
                        # Play the chunk out loud (asynchronously)
                        sd.play(audio_chunk, samplerate=sample_rate)
                        
                    # Prepare for model
                    waveform = audio_chunk[None, :]  # (1, audio_length)
                    waveform = move_data_to_device(waveform, device)
                    
                    # Forward pass
                    with torch.no_grad():
                        batch_output_dict = model(waveform, None)
                    
                    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
                    sorted_indexes = np.argsort(clipwise_output)[::-1]
                    
                    # Clear screen and print results
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print('=== Real-time Audio Tagging ===')
                    print('Top 10 detected sounds:')
                    for k in range(10):
                        confidence = clipwise_output[sorted_indexes[k]]
                        label = np.array(labels)[sorted_indexes[k]]
                        bar_length = int(confidence * 40)
                        bar = '█' * bar_length
                        print(f'{label:30s} {confidence:.3f} {bar}')
                    
                    print('\nPress Ctrl+C to stop recording...')
                else:
                    sd.sleep(10)
        
        except KeyboardInterrupt:
            print('\n\nRecording stopped.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Audio Tagging')
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)
    parser.add_argument('--model_type', type=str, default='Cnn14')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--chunk_duration', type=float, default=2.0, 
                       help='Duration of audio chunk to process in seconds')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--anc', action='store_true', default=False, help='Enable active noise cancelling')
    parser.add_argument('--play_audio', action='store_true', default=False, help='Play processed audio chunks back through speakers')
    
    args = parser.parse_args()
    realtime_audio_tagging(args)
