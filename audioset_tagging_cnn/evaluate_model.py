import argparse
import os
import glob
import numpy as np
import scipy.io.wavfile
import openwakeword
from openwakeword.model import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

def load_audio(file_path):
    """Loads a wav file and ensures it's mono."""
    sr, audio = scipy.io.wavfile.read(file_path)
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]
        
    # Openwakeword models expect 16khz audio, 16-bit PCM arrays.
    # Note: If users have other sample rates, they should resample it to 16khz first.
    return audio

def evaluate_model(model_path, wakeword_name, positive_dir, negative_dir, threshold=0.5):
    print(f"Loading model from {model_path}...")
    openwakeword.utils.download_models()
    model = Model(wakeword_models=[model_path])
    
    if wakeword_name not in model.models:
        wakeword_name = list(model.models.keys())[0]
        print(f"Fallback to model label: {wakeword_name}")
        
    y_true = []
    y_pred = []
    y_scores = []
    
    def process_dir(directory, label):
        if not os.path.exists(directory):
            print(f"Warning: Directory '{directory}' does not exist.")
            return

        wav_files = glob.glob(os.path.join(directory, "*.wav"))
        print(f"Found {len(wav_files)} files in {directory} (Label: {label})")
        
        for file in wav_files:
            try:
                audio = load_audio(file)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
                
            # Reset model state for each file
            model.reset()
            
            # Predict step-by-step to find max score in the file
            chunk_size = 1280 # 80ms at 16khz
            max_score = 0.0
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    # Pad
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                preds = model.predict(chunk)
                score = preds.get(wakeword_name, 0.0)
                if score > max_score:
                    max_score = score
            
            y_true.append(label)
            y_scores.append(max_score)
            y_pred.append(1 if max_score >= threshold else 0)

    # Process positive samples (label = 1)
    if positive_dir:
        process_dir(positive_dir, 1)
    else:
        print("Warning: No positive directory provided.")
        
    # Process negative samples (label = 0)
    if negative_dir:
        process_dir(negative_dir, 0)
    else:
        print("Warning: No negative directory provided.")
        
    if len(y_true) == 0:
        print("No audio files processed. Please check the directories.")
        return
        
    # --- Calculate Metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n" + "="*40)
    print("OVERALL PERFORMANCE METRICS")
    print("="*40)
    print(f"Threshold: {threshold}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*40)
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, '{:d}'.format(z), ha='center', va='center')
        
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    ax.xaxis.set_ticks_position('bottom')
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved 'confusion_matrix.png'")
    
    # --- Confidence Distribution ---
    plt.figure(figsize=(8, 5))
    pos_scores = [score for true, score in zip(y_true, y_scores) if true == 1]
    neg_scores = [score for true, score in zip(y_true, y_scores) if true == 0]
    
    if len(pos_scores) > 0:
        plt.hist(pos_scores, bins=20, color='green', alpha=0.5, label='Positive Samples', density=True)
    if len(neg_scores) > 0:
        plt.hist(neg_scores, bins=20, color='red', alpha=0.5, label='Negative Samples', density=True)
        
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    plt.title('Confidence Score Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig('confidence_distribution.png')
    print("Saved 'confidence_distribution.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate openWakeWord model performance.")
    parser.add_argument("--model_path", required=True, help="Path to the custom wake word model (.onnx or .tflite)")
    parser.add_argument("--wakeword_name", default="custom", help="Model label name inside the model")
    parser.add_argument("--positive_dir", default="", help="Directory containing positive .wav files")
    parser.add_argument("--negative_dir", default="", help="Directory containing negative .wav files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0 to 1.0)")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.wakeword_name, args.positive_dir, args.negative_dir, args.threshold)
