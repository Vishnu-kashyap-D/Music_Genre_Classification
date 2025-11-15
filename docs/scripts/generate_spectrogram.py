"""Utility script to generate spectrogram figures for documentation."""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

SR = 22050
OUTPUT = Path("docs/images/mel_spectrogram_example.png")

CANDIDATES = [
    Path("tmp_short.wav"),
    Path("archive/Data/genres_original/blues/blues.00000.wav"),
    Path("genres_original/blues/blues.00000.wav"),
]

def load_signal():
    for cand in CANDIDATES:
        if cand.exists():
            signal, sr = librosa.load(cand, sr=SR, mono=True, duration=5.0)
            return signal, sr, cand

    duration = 5.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    synth = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    return synth.astype(np.float32), SR, None


def main():
    signal, sr, source = load_signal()
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(mel_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel")
    title = "Example Mel-Spectrogram (5 s clip)"
    if source is not None:
        title += f"\nSource: {source.name}"
    plt.title(title)
    plt.colorbar(format="%+2.0f dB", pad=0.01)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=200)
    print(f"Saved spectrogram figure to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
