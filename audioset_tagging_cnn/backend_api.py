import os
import sys
import asyncio
import json
import logging
import base64
import numpy as np

# Add pytorch/ and utils/ to sys.path so models, config, pytorch_utils can be imported
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(_SCRIPT_DIR, 'pytorch'))
sys.path.insert(2, os.path.join(_SCRIPT_DIR, 'utils'))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa

# Suppress some logging for cleaner output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend_api")

try:
    from models import *
    from pytorch_utils import move_data_to_device
    import config
except ImportError:
    pass # Will handle relative imports dynamically if needed

import noisereduce as nr

# openwakeword is optional — audio tagging works without it
_HAS_OPENWAKEWORD = False
try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
    _HAS_OPENWAKEWORD = True
except ImportError:
    logger.warning("openwakeword not installed. Wake word detection will be unavailable.")
    logger.warning("Install it with: pip install openwakeword")

app = FastAPI(title="Audio Pattern Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------- #
# AUDIO TAGGING SETUP
# ---------------------------- #
# Hardcoded to Cnn14 default from scripts
AT_MODEL_TYPE = "Cnn14"
AT_CHECKPOINT = "Cnn14_mAP=0.431.pth"
AT_SAMPLE_RATE = 32000

at_model = None
at_device = None
at_labels = []

def init_audio_tagging():
    global at_model, at_device, at_labels
    
    abs_checkpoint = os.path.abspath(AT_CHECKPOINT)
    logger.info(f"Looking for checkpoint at: {abs_checkpoint}")
    
    if not os.path.exists(AT_CHECKPOINT):
        logger.warning(f"Checkpoint {abs_checkpoint} not found. Audio tagging may fail.")
        return
        
    try:
        import config
        from models import Cnn14
        from pytorch_utils import move_data_to_device
    except ImportError as e:
        logger.error(f"Could not import models or config: {e}")
        logger.error(f"sys.path = {sys.path}")
        return

    at_labels = config.labels
    classes_num = config.classes_num
    
    at_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    at_model = Cnn14(sample_rate=AT_SAMPLE_RATE, window_size=1024, 
        hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
        classes_num=classes_num)
    
    checkpoint = torch.load(AT_CHECKPOINT, map_location=at_device)
    at_model.load_state_dict(checkpoint['model'])
    at_model.eval()

    if 'cuda' in str(at_device):
        at_model.to(at_device)
        logger.info(f"Audio Tagging model loaded on GPU")
        at_model = torch.nn.DataParallel(at_model)
    else:
        logger.info("Audio Tagging model loaded on CPU.")

# ---------------------------- #
# WAKE WORD SETUP
# ---------------------------- #
WW_MODEL_PATH = "shiv_aak_aash.onnx"
ww_model = None

def init_wake_word():
    global ww_model
    if not _HAS_OPENWAKEWORD:
        logger.warning("openwakeword not available, skipping wake word init.")
        return
    if not os.path.exists(WW_MODEL_PATH):
        logger.warning(f"Wake word model {WW_MODEL_PATH} not found. Wake word may fail.")
        return
    openwakeword.utils.download_models()
    ww_model = OWWModel(wakeword_models=[WW_MODEL_PATH])
    logger.info("Wake word model loaded.")

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing models...")
    # Change cwd to script dir temporarily if needed
    original_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    init_audio_tagging()
    init_wake_word()
    os.chdir(original_cwd)
    logger.info("Models initialized.")

@app.websocket("/ws/audio_tagging")
async def ws_audio_tagging(websocket: WebSocket, anc: bool = False):
    await websocket.accept()
    logger.info(f"WebSocket connected: Audio Tagging (ANC: {anc})")
    
    chunk_samples = int(2.0 * AT_SAMPLE_RATE)
    step_samples = int(0.5 * AT_SAMPLE_RATE)
    audio_buffer = np.array([], dtype=np.float32)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            # Expecting float32 raw audio stream from browser WebAudio API
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, audio_chunk))
            
            if len(audio_buffer) >= chunk_samples:
                # Take latest 2s chunk for processing
                process_buf = audio_buffer[:chunk_samples]
                # Advance buffer by step size to keep rolling window
                audio_buffer = audio_buffer[step_samples:]
                
                if anc:
                    process_buf = nr.reduce_noise(y=process_buf, sr=AT_SAMPLE_RATE)
                
                if at_model is not None:
                    waveform = process_buf[None, :]
                    from pytorch_utils import move_data_to_device
                    waveform_tensor = move_data_to_device(waveform, at_device)
                    
                    with torch.no_grad():
                        batch_output_dict = at_model(waveform_tensor, None)
                    
                    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
                    sorted_indexes = np.argsort(clipwise_output)[::-1]
                    
                    results = []
                    for k in range(10):
                        confidence = float(clipwise_output[sorted_indexes[k]])
                        label = str(at_labels[sorted_indexes[k]])
                        results.append({"label": label, "score": confidence})
                    
                    await websocket.send_text(json.dumps({"type": "tagging_results", "data": results}))
                else:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Model not loaded"}))
                
    except (WebSocketDisconnect, Exception) as e:
        if not isinstance(e, WebSocketDisconnect):
            logger.debug(f"WebSocket send error (Audio Tagging): {type(e).__name__}")
        logger.info("WebSocket disconnected: Audio Tagging")

@app.websocket("/ws/wake_word")
async def ws_wake_word(websocket: WebSocket, anc: bool = False, expected_wakeword: str = "shiv_aak_aash", threshold: float = 0.5):
    await websocket.accept()
    logger.info(f"WebSocket connected: Wake Word '{expected_wakeword}' (ANC: {anc})")
    
    sample_rate = 16000 # OpenWakeWord expects 16kHz
    frame_samples = 1280
    audio_buffer = np.array([], dtype=np.int16)
    
    try:
        while True:
            data = await websocket.receive_bytes()
            # Browser sends 16kHz float32 audio
            audio_chunk_float = np.frombuffer(data, dtype=np.float32)
            audio_chunk = (audio_chunk_float * 32768).astype(np.int16)
            
            audio_buffer = np.concatenate((audio_buffer, audio_chunk))
            
            while len(audio_buffer) >= frame_samples:
                process_buf = audio_buffer[:frame_samples]
                audio_buffer = audio_buffer[frame_samples:]
                
                if anc:
                    process_buf = nr.reduce_noise(y=process_buf, sr=sample_rate)
                    
                if ww_model is not None:
                    preds = ww_model.predict(process_buf)
                    score = 0.0
                    if expected_wakeword in preds:
                        score = float(preds[expected_wakeword])
                    else:
                        if len(preds) > 0:
                            score = float(list(preds.values())[0])
                    
                    try:
                        if score >= threshold:
                            await websocket.send_text(json.dumps({
                                "type": "wake_word_detected", 
                                "data": {"score": score, "wakeword": expected_wakeword}
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "wake_word_score",
                                "data": {"score": score}
                            }))
                    except Exception:
                        break
                else:
                    try:
                        await websocket.send_text(json.dumps({"type": "error", "message": "Model not loaded"}))
                    except Exception:
                        break
                    
                await asyncio.sleep(0.001)
                
    except (WebSocketDisconnect, Exception) as e:
        if not isinstance(e, WebSocketDisconnect):
            logger.debug(f"WebSocket send error (Wake Word): {type(e).__name__}")
        logger.info("WebSocket disconnected: Wake Word")
