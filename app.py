"""
Flask Backend for Music Genre Classification Web App
Serves the Parallel CNN model with segmentation and analysis features.
"""

import os
import base64
import io
import hashlib
import math
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from train_parallel_cnn import (
    DatasetConfig,
    OpenL3Config,
    ParallelCNN,
    choose_device,
    load_openl3_model,
    compute_mel_slices,
)

app = Flask(__name__)
# Enhanced CORS configuration for better frontend-backend communication
CORS(app, 
     origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type"],
     supports_credentials=True)

# Configuration
MODEL_PATH = os.path.join("torch_models", "parallel_genre_classifier_torch.pt")
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "m4a", "ogg"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_ANALYSIS_DURATION = 30.0  # seconds of audio analyzed per request
REQUEST_TIMEOUT = 300  # 5 minutes for processing

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variables (loaded on first request)
_model = None
_mapping = None
_device = None
_cfg = None
_feature_type = None
_l3_config = None
_l3_model = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the PyTorch model and configuration."""
    global _model, _mapping, _device, _cfg, _feature_type, _l3_config, _l3_model
    
    if _model is not None:
        return  # Already loaded
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    _device = choose_device(None, False)
    payload = torch.load(MODEL_PATH, map_location=_device)
    _mapping = payload["mapping"]
    meta = payload.get("meta", {})
    
    # Load original config from checkpoint
    _cfg = DatasetConfig(
        window_duration=int(meta.get("window_duration", 15)),
        slice_duration=int(meta.get("slice_duration", 5)),
        window_stride=int(meta.get("window_stride", 5)),
    )
    
    _feature_type = str(meta.get("feature_type", "mel"))
    _l3_config = None
    _l3_model = None
    input_feature_dim = None
    
    if _feature_type == "openl3":
        try:
            openl3_meta = meta.get("openl3", {})
            _l3_config = OpenL3Config(
                embedding_dim=int(openl3_meta.get("embedding_dim", 512)),
                content_type=str(openl3_meta.get("content_type", "music")),
                input_repr=str(openl3_meta.get("input_repr", "mel256")),
                center=bool(openl3_meta.get("center", False)),
                hop_size=float(openl3_meta.get("hop_size", 0.5)),
                batch_size=int(openl3_meta.get("batch_size", 32)),
            )
            input_feature_dim = _l3_config.embedding_dim
            _l3_model = load_openl3_model(_l3_config)
        except (ImportError, RuntimeError) as e:
            print(f"WARNING: OpenL3 not available: {e}")
            print("Falling back to mel features. Install OpenL3 if you need openl3 feature type.")
            print("See INSTALL_OPENL3.md for installation instructions.")
            # Fallback to mel if OpenL3 is not available
            _feature_type = "mel"
    
    _model = ParallelCNN(
        num_slices=_cfg.slices_per_window,
        num_classes=len(_mapping),
        embedding_dim=int(meta.get("embedding_dim", 160)),
        shared_backbone=bool(meta.get("shared_backbone", True)),
        feature_type=_feature_type,
        input_feature_dim=input_feature_dim,
    )
    _model.load_state_dict(payload["state_dict"])
    _model.to(_device)
    _model.eval()
    
    print(f"Model loaded successfully on {_device}")
    print(f"Feature type: {_feature_type}, Classes: {len(_mapping)}")


def check_audio_quality(signal: np.ndarray, sample_rate: int = 22050) -> Tuple[bool, Optional[str]]:
    """
    Check if audio has sufficient signal quality and is music-like (not noise/traffic).
    Returns (is_valid, error_message).
    
    Checks:
    1. RMS energy (silent/quiet detection)
    2. SNR (signal-to-noise ratio)
    3. Spectral features (to detect non-musical sounds like traffic)
    4. Frequency distribution (to detect high-pitch mechanical sounds)
    """
    if len(signal) == 0:
        return False, "Audio file is empty or could not be read."
    
    # Check 1: RMS energy (silent/quiet audio)
    rms_energy = np.sqrt(np.mean(signal ** 2))
    RMS_THRESHOLD = 0.005
    
    if rms_energy < RMS_THRESHOLD:
        return False, "Audio is too quiet or silent. Please upload a clearer track with audible content."
    
    # Check 2: SNR (noisy audio detection) - but only reject if extremely noisy
    snr = compute_snr(signal)
    SNR_THRESHOLD = 3.0  # dB - lowered threshold, only reject extremely noisy audio
    
    # Only reject if SNR is very low AND RMS is reasonable (indicates noise, not silence)
    if snr < SNR_THRESHOLD and rms_energy > 0.01:
        print(f"REJECTED: Very low SNR detected ({snr:.1f}dB) with reasonable RMS ({rms_energy:.4f})")
        return False, "Audio is too noisy or contains excessive background noise. Please upload a cleaner recording."
    
    # Check 3: Spectral analysis for non-musical sounds
    try:
        # Extract spectral features
        frame_length = 2048
        hop_length = 512
        
        # Spectral centroid (brightness) - traffic/mechanical sounds have very high centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
        )[0]
        avg_centroid = np.mean(spectral_centroid)
        max_centroid = np.max(spectral_centroid)
        
        # Zero crossing rate (ZCR) - noise has high ZCR, music has moderate ZCR
        zcr = librosa.feature.zero_crossing_rate(
            y=signal, frame_length=frame_length, hop_length=hop_length
        )[0]
        avg_zcr = np.mean(zcr)
        max_zcr = np.max(zcr)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
        )[0]
        avg_rolloff = np.mean(spectral_rolloff)
        max_rolloff = np.max(spectral_rolloff)
        
        # Spectral bandwidth (spread of frequencies)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
        )[0]
        avg_bandwidth = np.mean(spectral_bandwidth)
        bandwidth_std = np.std(spectral_bandwidth)
        
        # Spectral contrast (difference between peaks and valleys)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=signal, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
        )
        avg_contrast = np.mean(spectral_contrast)
        
        # Log spectral features for debugging
        print(f"Audio Quality Metrics - Centroid: {avg_centroid:.1f}Hz (max: {max_centroid:.1f}Hz), "
              f"ZCR: {avg_zcr:.3f} (max: {max_zcr:.3f}), Rolloff: {avg_rolloff:.1f}Hz (max: {max_rolloff:.1f}Hz), "
              f"SNR: {snr:.1f}dB")
        
        # Check 3a: High-frequency noise (traffic sounds, mechanical sounds)
        # Traffic sounds typically have very high spectral centroid (>5500 Hz) and high ZCR
        HIGH_FREQ_THRESHOLD = 5500  # Hz - lowered to catch more traffic sounds
        HIGH_ZCR_THRESHOLD = 0.12  # Lowered threshold for better detection
        
        if avg_centroid > HIGH_FREQ_THRESHOLD and avg_zcr > HIGH_ZCR_THRESHOLD:
            print(f"REJECTED: High-frequency noise detected (centroid: {avg_centroid:.1f}Hz, ZCR: {avg_zcr:.3f})")
            return False, "Audio appears to contain non-musical sounds (e.g., traffic, mechanical noise). Please upload a music track."
        
        # Check 3b: Very high centroid alone (strong indicator of traffic/mechanical sounds)
        if avg_centroid > 7000:  # Very high frequency content
            print(f"REJECTED: Very high spectral centroid detected ({avg_centroid:.1f}Hz)")
            return False, "Audio contains high-frequency non-musical sounds (e.g., traffic, sirens). Please upload a music track."
        
        # Check 3c: Excessive high-frequency content (high-pitch traffic, sirens)
        # Music typically has rolloff between 2000-8000 Hz, traffic can exceed 9000 Hz
        if avg_rolloff > 9000 and avg_centroid > 4500:  # Lowered thresholds
            print(f"REJECTED: Excessive high-frequency content (rolloff: {avg_rolloff:.1f}Hz, centroid: {avg_centroid:.1f}Hz)")
            return False, "Audio contains high-pitch non-musical sounds. Please upload a music track for genre classification."
        
        # Check 3d: Very high rolloff alone (indicates excessive high frequencies)
        if max_rolloff > 11000:  # Very high rolloff indicates sirens/high-pitch sounds
            print(f"REJECTED: Very high spectral rolloff detected ({max_rolloff:.1f}Hz)")
            return False, "Audio contains very high-pitch sounds (e.g., sirens, alarms). Please upload a music track."
        
        # Check 3e: Very noisy audio (high ZCR with low spectral structure)
        # Music has structured spectral content, pure noise has high ZCR and low bandwidth variation
        zcr_std = np.std(zcr)
        if avg_zcr > 0.10 and zcr_std < 0.015 and bandwidth_std < 400:  # Stricter thresholds
            print(f"REJECTED: Noise/static detected (ZCR: {avg_zcr:.3f}, low variation)")
            return False, "Audio appears to be noise or static rather than music. Please upload a clear music track."
        
        # Check 3f: Very flat spectral content (white noise, static)
        # Music has varying spectral content over time
        centroid_std = np.std(spectral_centroid)
        if centroid_std < 150 and avg_zcr > 0.08:  # Stricter thresholds
            print(f"REJECTED: Flat spectral content detected (centroid_std: {centroid_std:.1f}Hz, ZCR: {avg_zcr:.3f})")
            return False, "Audio lacks musical structure. Please upload a music track with clear musical content."
        
        # Check 3g: Low spectral contrast (indicates noise rather than music)
        # Music has higher spectral contrast (clear peaks and valleys)
        if avg_contrast < 2.0 and avg_zcr > 0.09:  # Low contrast + high ZCR = noise
            print(f"REJECTED: Low spectral contrast detected (contrast: {avg_contrast:.2f}, ZCR: {avg_zcr:.3f})")
            return False, "Audio lacks musical structure and appears to be noise. Please upload a clear music track."
        
        # Check 3h: Combination check - high ZCR with high centroid (strong noise indicator)
        if avg_zcr > 0.13 and avg_centroid > 5000:
            print(f"REJECTED: High ZCR + high centroid combination (ZCR: {avg_zcr:.3f}, centroid: {avg_centroid:.1f}Hz)")
            return False, "Audio appears to contain non-musical sounds or excessive noise. Please upload a music track."
        
    except Exception as e:
        # If spectral analysis fails, fall back to basic checks
        print(f"Warning: Spectral analysis failed: {e}. Using basic quality checks only.")
        import traceback
        traceback.print_exc()
        pass
    
    return True, None


def compute_snr(signal: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.
    Uses a simple approach: signal power vs estimated noise floor.
    """
    if len(signal) == 0:
        return 0.0
    
    # Estimate noise as the minimum energy in small windows
    window_size = min(2048, len(signal) // 10)
    if window_size < 100:
        return 20.0  # Default for very short signals
    
    noise_estimate = np.inf
    for i in range(0, len(signal) - window_size, window_size):
        window_energy = np.mean(signal[i:i+window_size] ** 2)
        noise_estimate = min(noise_estimate, window_energy)
    
    signal_power = np.mean(signal ** 2)
    if noise_estimate <= 0 or signal_power <= 0:
        return 20.0
    
    snr_db = 10 * np.log10(signal_power / noise_estimate)
    return float(np.clip(snr_db, 0, 60))  # Clamp to reasonable range


def prepare_audio_for_segmentation(signal: np.ndarray, sample_rate: int, target_duration: float = 30.0) -> np.ndarray:
    """
    Prepare audio for segmentation by padding or looping if too short.
    
    If audio is shorter than target_duration:
    - For music: Loop the audio to fill the time (preferred)
    - Alternative: Pad with silence (zeros)
    
    Args:
        signal: Audio signal array
        sample_rate: Sample rate in Hz
        target_duration: Target duration in seconds (default 30s)
    
    Returns:
        Prepared audio signal with at least target_duration seconds
    """
    current_duration = len(signal) / sample_rate
    
    if current_duration >= target_duration:
        # Audio is long enough, return as-is (truncate to target if needed)
        target_samples = int(target_duration * sample_rate)
        if len(signal) > target_samples:
            return signal[:target_samples]
        return signal
    
    # Audio is too short - loop it to fill the time
    target_samples = int(target_duration * sample_rate)
    num_loops = int(np.ceil(target_samples / len(signal)))
    
    # Loop the audio
    looped_signal = np.tile(signal, num_loops)
    
    # Trim to exact target length
    if len(looped_signal) > target_samples:
        looped_signal = looped_signal[:target_samples]
    
    print(f"Audio was {current_duration:.2f}s, looped to {target_duration:.2f}s for processing")
    return looped_signal


def process_3s_segments(
    signal: np.ndarray,
    sample_rate: int,
    num_segments: int = 10
) -> List[np.ndarray]:
    """
    Process audio into 3-second segments and prepare for model inference.
    Supports both mel and openl3 feature types based on model configuration.
    Adapts 3s segments to work with model that expects 3 x 5s slices.
    
    Now handles short audio gracefully - works with any number of segments >= 1.
    """
    segment_duration = 3.0  # seconds
    segment_samples = int(sample_rate * segment_duration)
    
    # Calculate how many segments we can actually create from the audio
    audio_duration = len(signal) / sample_rate
    max_possible_segments = max(1, int(np.floor(audio_duration / segment_duration)))
    
    # Use the minimum of requested segments and possible segments
    actual_num_segments = min(num_segments, max_possible_segments)
    
    if actual_num_segments < num_segments:
        print(f"Warning: Requested {num_segments} segments but only {actual_num_segments} fit in {audio_duration:.2f}s audio")
    
    segments = []
    
    for i in range(actual_num_segments):
        start_sample = int(i * segment_samples)
        end_sample = start_sample + segment_samples
        
        if start_sample >= len(signal):
            # This shouldn't happen with our adaptive logic, but handle it gracefully
            break
        elif end_sample > len(signal):
            # Last segment - pad if needed
            seg = signal[start_sample:].copy()
            seg = librosa.util.fix_length(seg, size=segment_samples)
        else:
            seg = signal[start_sample:end_sample].copy()
        
        # Pad each 3s segment to 5s to match model's expected slice duration
        slice_5s = librosa.util.fix_length(seg, size=int(sample_rate * 5.0))
        
        # Process based on feature type
        if _feature_type == "mel":
            mel = librosa.feature.melspectrogram(
                y=slice_5s,
                sr=sample_rate,
                n_mels=_cfg.n_mels,
                n_fft=_cfg.n_fft,
                hop_length=_cfg.hop_length
            )
            mel_db = librosa.power_to_db(mel, ref=np.max, top_db=None)
            mel_db = pad_or_truncate(mel_db, _cfg.expected_frames).astype(np.float32)
            # Repeat same slice to satisfy model's expected shape (3, n_mels, time_frames)
            segment_tensor = np.repeat(mel_db[None, ...], 3, axis=0)
            segments.append(segment_tensor)
            
        elif _feature_type == "openl3":
            if _l3_model is None or _l3_config is None:
                raise RuntimeError("OpenL3 model not loaded. Cannot process with openl3 feature type.")
            
            try:
                import openl3
            except ImportError:
                raise RuntimeError("OpenL3 package not installed. Install with: pip install openl3")
            
            embeddings, _ = openl3.get_audio_embedding(
                slice_5s,
                sr=sample_rate,
                hop_size=_l3_config.hop_size,
                center=_l3_config.center,
                batch_size=_l3_config.batch_size,
                model=_l3_model,
            )
            if embeddings.size == 0:
                slice_emb = np.zeros(_l3_config.embedding_dim, dtype=np.float32)
            else:
                slice_emb = embeddings.mean(axis=0).astype(np.float32)
            segment_tensor = np.tile(slice_emb, (3, 1))
            segments.append(segment_tensor)
        else:
            raise ValueError(f"Unsupported feature type: {_feature_type}")
    
    return segments


def pad_or_truncate(mel_db: np.ndarray, expected_frames: int) -> np.ndarray:
    """Pad or truncate mel-spectrogram to expected frames."""
    if mel_db.shape[1] == expected_frames:
        return mel_db
    if mel_db.shape[1] < expected_frames:
        pad_width = expected_frames - mel_db.shape[1]
        return np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    return mel_db[:, :expected_frames]


def generate_spectrogram_image(signal: np.ndarray, sample_rate: int) -> str:
    """
    Generate mel-spectrogram image and return as base64 string.
    """
    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    img = ax.imshow(
        mel_db,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='bilinear'
    )
    ax.set_title('Mel-Spectrogram', fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel('Time (frames)', fontsize=12, color='white')
    ax.set_ylabel('Mel Frequency Bins', fontsize=12, color='white')
    
    # Dark theme styling
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='white', labelsize=10)
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    plt.colorbar(img, ax=ax, label='dB')
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1a1a1a', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        model_status = _model is not None
        if not model_status:
            # Try to load model if not loaded
            try:
                load_model()
                model_status = True
            except Exception as e:
                return jsonify({
                    "status": "degraded", 
                    "model_loaded": False,
                    "error": str(e)
                }), 503
        
        return jsonify({
            "status": "healthy", 
            "model_loaded": model_status,
            "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
            "max_analysis_duration": MAX_ANALYSIS_DURATION
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    Accepts audio file and returns genre predictions with analysis.
    """
    temp_path = None
    try:
        # Load model if not already loaded
        print("Loading model...")
        load_model()
        print("Model loaded successfully")
        
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        # Check file size before saving
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                "error": f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.1f}MB, got {file_size / (1024 * 1024):.1f}MB"
            }), 400
        
        if file_size == 0:
            return jsonify({"error": "File is empty"}), 400
        
        print(f"Processing file: {file.filename} ({file_size / (1024 * 1024):.2f}MB)")
        
        # Save uploaded file temporarily
        file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
        temp_path = os.path.join(UPLOAD_FOLDER, f"{file_hash}_{file.filename}")
        file.save(temp_path)
        print(f"File saved to: {temp_path}")
        
        try:
            # Load audio with robust error handling for corrupt/invalid files
            print("Loading audio file...")
            original_duration = None
            try:
                audio_info = sf.info(temp_path)
                original_duration = float(audio_info.duration)
                print(f"Audio duration: {original_duration:.2f} seconds")
            except Exception as duration_error:
                print(f"Warning: Unable to read audio duration metadata: {duration_error}")
            
            print("Extracting audio signal...")
            try:
                signal, sr = librosa.load(
                    temp_path,
                    sr=_cfg.sample_rate,
                    mono=True,
                    duration=MAX_ANALYSIS_DURATION
                )
            except Exception as load_error:
                # Handle various librosa/soundfile errors
                error_type = type(load_error).__name__
                error_msg = str(load_error).lower()
                
                if "nobackend" in error_msg or "backend" in error_msg:
                    return jsonify({
                        "error": "Invalid audio format. Please upload a valid MP3, WAV, FLAC, M4A, or OGG file."
                    }), 400
                elif "could not read" in error_msg or "unable to" in error_msg:
                    return jsonify({
                        "error": "Invalid audio format. The file appears to be corrupted or not a valid audio file. Please upload a valid MP3 or WAV file."
                    }), 400
                elif "format" in error_msg:
                    return jsonify({
                        "error": "Invalid audio format. Please upload a valid MP3 or WAV file."
                    }), 400
                else:
                    return jsonify({
                        "error": f"Invalid audio format. Could not read audio file: {str(load_error)}. Please upload a valid MP3 or WAV file."
                    }), 400
            
            # Check if signal was loaded successfully
            if signal is None or len(signal) == 0:
                return jsonify({
                    "error": "Invalid audio format. The file appears to be empty or corrupted. Please upload a valid audio file."
                }), 400
            
            duration = len(signal) / sr
            if original_duration is None:
                original_duration = duration
            
            print(f"Loaded {duration:.2f} seconds of audio (sample rate: {sr}Hz)")
            
            # Fix 1: Check for silent/quiet audio, noise, and non-musical sounds
            print("Checking audio quality (RMS, SNR, and spectral analysis)...")
            is_valid, quality_error = check_audio_quality(signal, sr)
            if not is_valid:
                return jsonify({"error": quality_error}), 400
            
            # Fix 2: Handle short audio - prepare it for segmentation
            # Minimum 3 seconds required for at least one segment
            if duration < 3.0:
                return jsonify({
                    "error": "Audio too short. Minimum 3 seconds required for analysis."
                }), 400
            
            # Prepare audio: if shorter than 30s, loop it to fill the time
            # This ensures we can create up to 10 segments even for short clips
            target_duration = min(MAX_ANALYSIS_DURATION, 30.0)  # Target 30s for 10 segments
            if duration < target_duration:
                signal = prepare_audio_for_segmentation(signal, sr, target_duration)
                duration = len(signal) / sr  # Update duration after looping
            
            # Calculate SNR
            print("Calculating SNR...")
            snr = compute_snr(signal)
            
            # Process into 3-second segments (adaptive - works with any number of segments)
            requested_segments = max(1, min(10, math.ceil(duration / 3.0)))
            print(f"Processing up to {requested_segments} segments from {duration:.2f}s audio...")
            segments = process_3s_segments(signal, sr, requested_segments)
            
            if not segments:
                return jsonify({"error": "Failed to process audio segments"}), 500
            
            # Get actual number of segments processed (may be less than requested for short audio)
            actual_num_segments = len(segments)
            
            # Run inference on all segments
            print("Running model inference...")
            all_predictions = []
            segment_predictions = []
            
            with torch.inference_mode():
                for seg_idx, segment_tensor in enumerate(segments):
                    print(f"Processing segment {seg_idx + 1}/{actual_num_segments}...")
                    # Convert to tensor: shape (1, 3, n_mels, time_frames)
                    tensor = torch.from_numpy(segment_tensor[None, ...]).to(_device)
                    
                    # Predict
                    output = _model(tensor)
                    probs = torch.sigmoid(output).cpu().numpy()[0]  # Shape: (num_classes,)
                    
                    all_predictions.append(probs)
                    
                    # Get top genre for this segment
                    top_idx = np.argmax(probs)
                    top_genre = _mapping[top_idx]
                    confidence = float(probs[top_idx])
                    
                    segment_predictions.append({
                        "start": seg_idx * 3.0,
                        "end": (seg_idx + 1) * 3.0,
                        "top_genre": top_genre,
                        "confidence": confidence,
                        "all_probs": {_mapping[i]: float(probs[i]) for i in range(len(_mapping))}
                    })
            
            print("Computing global predictions...")
            # Compute global confidence (average across all segments)
            global_probs = np.mean(all_predictions, axis=0)
            global_confidence = {
                _mapping[i]: float(global_probs[i]) for i in range(len(_mapping))
            }
            
            # Generate spectrogram image
            print("Generating spectrogram...")
            spectrogram_image = generate_spectrogram_image(signal, sr)
            
            # Build response
            print("Building response...")
            response = {
                "global_confidence": global_confidence,
                "timeline": segment_predictions,
                "spectrogram_image": spectrogram_image,
                "metrics": {
                    "snr": snr,
                    "duration": float(duration),
                    "original_duration": float(original_duration),
                    "analysis_window": MAX_ANALYSIS_DURATION,
                    "num_segments": actual_num_segments
                },
                "filename": file.filename
            }
            
            print("Analysis complete!")
            return jsonify(response)
        
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"Error during processing: {error_msg}")
            traceback.print_exc()
            return jsonify({"error": f"Processing failed: {error_msg}"}), 500
        
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not delete temp file: {cleanup_error}")
    
    except FileNotFoundError as e:
        return jsonify({"error": f"Model file not found: {str(e)}. Please ensure the model is trained and available."}), 503
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Unexpected error: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {error_msg}"}), 500


if __name__ == '__main__':
    print("Starting Flask server...")
    print("Loading model on startup...")
    try:
        load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

