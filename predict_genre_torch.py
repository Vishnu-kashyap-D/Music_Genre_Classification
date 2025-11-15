import argparse
import json
import os
import importlib
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn

librosa_utils = importlib.import_module("librosa.util.utils")

MODEL_PATH = os.path.join("torch_models", "genre_classifier_torch.pt")
POP_HIPHOP_AUX_PATH = os.path.join("torch_models", "pop_hiphop_aux.joblib")
SAMPLE_RATE = 22050
DURATION_SECONDS = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION_SECONDS
NUM_SEGMENTS = 10
SAMPLES_PER_SEGMENT = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
EXPECTED_VECTORS_PER_SEGMENT = 130
LOG_PATH = Path("logs/rejections.log")


@dataclass
class PrecheckConfig:
    min_duration_sec: float = 2.0
    silence_rms_threshold: float = 1e-4
    silence_peak_threshold: float = 1e-3
    speech_fraction_threshold: float = 0.7
    overlap_speech_fraction: float = 0.35
    overlap_music_centroid: float = 1800.0
    low_snr_db_threshold: float = 6.0
    live_flatness_ratio: float = 0.6
    live_snr_ceiling_db: float = 10.0
    clipping_fraction_threshold: float = 0.02
    spike_quantile: float = 0.997
    spike_fraction_threshold: float = 0.01
    min_valid_sr: int = 12000
    max_valid_sr: int = 96000


CONFIG = PrecheckConfig()

REJECTION_MESSAGES: Dict[str, str] = {
    "low_snr": "Recording too noisy; provide clearer audio.",
    "speech_only": "Speech-only audio detected; please supply a music track.",
    "speech_music_overlap": "Speech dominates the track; provide music without vocals.",
    "live_acoustics": "Live or ambient recording detected; provide a studio-quality track.",
    "clipping": "Audio is clipped/distorted; submit a cleaner recording.",
    "spikes": "Audio contains spikes; check the source file and try again.",
    "sr_anomaly": "Unsupported sample rate detected; use standard 22.05-96 kHz audio.",
    "corrupt": "Could not read the audio file; make sure it isn't corrupted.",
    "classifier_error": "Classifier encountered an internal error; retry with a different file.",
}


class CNNGenreClassifier(nn.Module):
    """Mirror of the architecture used for training."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def log_rejection(file_path: str, reason: str, metrics: Dict[str, float]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "file_path": os.path.abspath(file_path),
        "reason": reason,
        "metrics": metrics,
    }
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_audio_safe(file_path: str, target_sr: int) -> Tuple[bool, np.ndarray | None, int | None, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    try:
        raw_signal, original_sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as exc:
        metrics["error"] = str(exc)
        return False, None, None, metrics

    if raw_signal.size == 0 or np.any(np.isnan(raw_signal)) or np.any(np.isinf(raw_signal)):
        metrics["issue"] = "invalid_samples"
        return False, None, None, metrics

    metrics["original_sr"] = float(original_sr)

    if original_sr != target_sr:
        signal = librosa.resample(raw_signal, orig_sr=original_sr, target_sr=target_sr)
    else:
        signal = raw_signal

    signal = signal.astype(np.float32, copy=False)
    return True, signal, original_sr, metrics


def estimate_speech_fraction(signal: np.ndarray, sr: int) -> float:
    frame_length = 2048
    hop_length = 512
    zcr = librosa.feature.zero_crossing_rate(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr,
        n_fft=frame_length,
        hop_length=hop_length,
    )[0]
    speech_frames = (zcr > 0.08) & (centroid < 4500)
    return float(np.mean(speech_frames))


def estimate_snr(signal: np.ndarray) -> float:
    frame_length = 2048
    hop_length = 512
    rms_frames = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]
    noise_floor = float(np.percentile(rms_frames, 25)) + 1e-8
    signal_level = float(np.percentile(rms_frames, 90)) + 1e-8
    snr_db = 20.0 * np.log10(signal_level / noise_floor)
    return snr_db


def estimate_music_ratio(signal: np.ndarray) -> float:
    harmonic, _ = librosa.effects.hpss(signal)
    energy_total = float(np.sum(signal ** 2)) + 1e-8
    energy_harmonic = float(np.sum(harmonic ** 2))
    return energy_harmonic / energy_total


def live_concert_flatness(signal: np.ndarray) -> float:
    flatness = librosa.feature.spectral_flatness(
        y=signal,
        n_fft=2048,
        hop_length=512,
    )[0]
    return float(np.mean(flatness > 0.5))


def detect_spike_fraction(signal: np.ndarray, config: PrecheckConfig) -> float:
    amplitude = np.abs(signal)
    threshold = np.quantile(amplitude, config.spike_quantile)
    if threshold <= 0:
        return 0.0
    return float(np.mean(amplitude > threshold))


def prepare_model_input(signal: np.ndarray) -> np.ndarray:
    if len(signal) < SAMPLES_PER_TRACK:
        signal = librosa_utils.fix_length(signal, size=SAMPLES_PER_TRACK)
    else:
        signal = signal[:SAMPLES_PER_TRACK]

    segments = []
    for segment in range(NUM_SEGMENTS):
        start = SAMPLES_PER_SEGMENT * segment
        stop = start + SAMPLES_PER_SEGMENT
        mel = librosa.feature.melspectrogram(
            y=signal[start:stop],
            sr=SAMPLE_RATE,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] == EXPECTED_VECTORS_PER_SEGMENT:
            segments.append(mel_db.T)
        elif mel_db.shape[1] < EXPECTED_VECTORS_PER_SEGMENT:
            pad_width = EXPECTED_VECTORS_PER_SEGMENT - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
            segments.append(mel_db.T)
        else:
            segments.append(mel_db[:, :EXPECTED_VECTORS_PER_SEGMENT].T)

    return np.stack(segments, axis=0)[:, None, :, :].astype(np.float32)


def compute_pop_hiphop_features(mel_batch: np.ndarray) -> np.ndarray | None:
    if mel_batch.ndim != 4 or mel_batch.shape[1] != 1:
        return None

    segments = mel_batch[:, 0, :, :]
    features = []
    for segment in segments:
        mean_band = segment.mean(axis=0)
        std_band = segment.std(axis=0)

        delta = np.diff(segment, axis=0, prepend=segment[:1])
        delta_mean = delta.mean(axis=0)
        delta_std = delta.std(axis=0)

        accel = np.diff(delta, axis=0, prepend=delta[:1])
        accel_mean = accel.mean(axis=0)
        accel_std = accel.std(axis=0)

        max_band = segment.max(axis=0)
        min_band = segment.min(axis=0)

        features.append(
            np.concatenate(
                [
                    mean_band,
                    std_band,
                    delta_mean,
                    delta_std,
                    accel_mean,
                    accel_std,
                    max_band,
                    min_band,
                ]
            )
        )

    if not features:
        return None

    return np.stack(features, axis=0)


@lru_cache(maxsize=1)
def load_pop_hiphop_aux() -> Dict[str, object] | None:
    if not os.path.exists(POP_HIPHOP_AUX_PATH):
        return None

    try:
        artifact = joblib.load(POP_HIPHOP_AUX_PATH)
    except Exception:
        return None

    if not isinstance(artifact, dict):
        return None

    model = artifact.get("model")
    scaler = artifact.get("scaler")
    threshold = float(artifact.get("threshold", 0.5))
    if model is None or scaler is None:
        return None

    return {"model": model, "scaler": scaler, "threshold": threshold}


def apply_pop_hiphop_resolver(
    averaged_probs: np.ndarray,
    mel_batch: np.ndarray,
    mapping: List[str],
) -> np.ndarray:
    aux = load_pop_hiphop_aux()
    if aux is None:
        return averaged_probs

    try:
        hiphop_idx = mapping.index("hiphop")
        pop_idx = mapping.index("pop")
    except ValueError:
        return averaged_probs

    hiphop_prob = averaged_probs[hiphop_idx]
    pop_prob = averaged_probs[pop_idx]
    combined_mass = hiphop_prob + pop_prob
    if combined_mass < 1e-6:
        return averaged_probs

    features = compute_pop_hiphop_features(mel_batch)
    if features is None:
        return averaged_probs

    scaler = aux["scaler"]
    model = aux["model"]
    threshold = aux["threshold"]

    standardized = scaler.transform(features)
    segment_probs = model.predict_proba(standardized)[:, 1]
    mean_prob = float(np.mean(segment_probs))

    adjusted = averaged_probs.copy()
    if mean_prob >= threshold:
        adjusted[hiphop_idx] = combined_mass
        adjusted[pop_idx] = 0.0
    else:
        adjusted[hiphop_idx] = 0.0
        adjusted[pop_idx] = combined_mass

    total = adjusted.sum()
    if total > 0:
        adjusted /= total
    return adjusted


def run_prechecks(signal: np.ndarray, original_sr: int, config: PrecheckConfig) -> Tuple[bool, str, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    duration = len(signal) / SAMPLE_RATE
    metrics["duration_sec"] = duration
    if duration < config.min_duration_sec:
        return False, "short_clip", metrics

    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))
    metrics["rms"] = rms
    metrics["peak"] = peak
    if rms < config.silence_rms_threshold and peak < config.silence_peak_threshold:
        metrics["silence_like"] = 1.0
        return False, "low_snr", metrics

    speech_fraction = estimate_speech_fraction(signal, SAMPLE_RATE)
    metrics["speech_fraction"] = speech_fraction

    snr_db = estimate_snr(signal)
    metrics["snr_db"] = snr_db
    if snr_db < config.low_snr_db_threshold:
        metrics["snr_reason"] = "below_threshold"

    music_ratio = estimate_music_ratio(signal)
    metrics["music_ratio"] = music_ratio
    if speech_fraction > config.speech_fraction_threshold and music_ratio < 0.2:
        metrics["speech_only"] = 1.0
        return False, "speech_only", metrics

    if speech_fraction > config.overlap_speech_fraction and music_ratio >= 0.2:
        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=signal,
                    sr=SAMPLE_RATE,
                )[0]
            )
        )
        metrics["spectral_centroid"] = centroid

    flat_ratio = live_concert_flatness(signal)
    metrics["flatness_ratio"] = flat_ratio
    if flat_ratio > config.live_flatness_ratio and snr_db < config.live_snr_ceiling_db:
        metrics["live_candidate"] = 1.0
        return False, "live_acoustics", metrics

    clipping_fraction = float(np.mean(np.abs(signal) > 0.98))
    metrics["clipping_fraction"] = clipping_fraction
    if clipping_fraction > config.clipping_fraction_threshold:
        return False, "clipping", metrics

    spike_fraction = detect_spike_fraction(signal, config)
    metrics["spike_fraction"] = spike_fraction
    if spike_fraction > config.spike_fraction_threshold:
        return False, "spikes", metrics

    metrics["original_sr"] = float(original_sr)
    if original_sr is None or original_sr < config.min_valid_sr or original_sr > config.max_valid_sr:
        return False, "sr_anomaly", metrics

    return True, "ok", metrics


def choose_device(gpu_index: int | None = None, force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch.cuda.is_available():
        print("⚙ Using CPU for inference")
        return torch.device("cpu")

    visible_devices = torch.cuda.device_count()
    if gpu_index is not None:
        if gpu_index < 0 or gpu_index >= visible_devices:
            print(f"⚠ GPU index {gpu_index} out of range; defaulting to cuda:0")
            gpu_index = 0
    else:
        gpu_index = 0

    torch.cuda.set_device(gpu_index)
    device = torch.device(f"cuda:{gpu_index}")
    name = torch.cuda.get_device_name(device)
    print(f"✓ Using CUDA device {gpu_index}: {name}")
    return device


def load_checkpoint(model_path: str, device: torch.device) -> tuple[CNNGenreClassifier, List[str]]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    payload = torch.load(model_path, map_location=device)
    mapping = payload["mapping"]
    model = CNNGenreClassifier(num_classes=len(mapping))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, mapping


def predict(file_path: str, model: CNNGenreClassifier, mapping: List[str], device: torch.device) -> None:
    ok, signal, original_sr, load_metrics = load_audio_safe(file_path, SAMPLE_RATE)
    if not ok:
        log_rejection(file_path, "corrupt", load_metrics)
        print(REJECTION_MESSAGES["corrupt"])
        return

    passes, reason, metrics = run_prechecks(signal, original_sr, CONFIG)
    if not passes:
        log_rejection(file_path, reason, {**load_metrics, **metrics})
        if reason == "short_clip":
            message = "Audio is too short."
        elif reason == "low_snr" and metrics.get("silence_like") == 1.0:
            message = "Silent audio detected."
        else:
            message = REJECTION_MESSAGES.get(reason, "Give proper audio input.")
        print(message)
        return

    data = prepare_model_input(signal)
    tensor = torch.from_numpy(data).to(device)

    print(f"\n--- Prediction for: {os.path.basename(file_path)} ---")
    try:
        with torch.inference_mode():
            model.eval()
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    except Exception as exc:
        log_rejection(file_path, "classifier_error", {**load_metrics, **metrics, "error": str(exc)})
        print(REJECTION_MESSAGES["classifier_error"])
        return

    averaged = probabilities.mean(axis=0)
    averaged = apply_pop_hiphop_resolver(averaged, data, mapping)
    top_indices = np.argsort(averaged)[::-1][:3]

    for idx in top_indices:
        genre = mapping[idx]
        prob = averaged[idx] * 100
        print(f"{genre.capitalize()}: {prob:.2f}%")
    print("------------------------------------------")


def main():
    parser = argparse.ArgumentParser(description="Predict music genre using the PyTorch model")
    parser.add_argument("audio_path", type=str, help="Path to the audio file to classify")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the PyTorch checkpoint (.pt)")
    parser.add_argument("--gpu-index", type=int, help="CUDA device index to use")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    device = choose_device(args.gpu_index, args.cpu)
    model, mapping = load_checkpoint(args.model, device)
    predict(args.audio_path, model, mapping, device)


if __name__ == "__main__":
    main()
