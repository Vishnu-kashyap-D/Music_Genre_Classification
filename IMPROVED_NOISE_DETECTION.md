# Improved Noise and Traffic Sound Detection

## Summary
Enhanced the audio quality detection with more comprehensive checks, better thresholds, and debug logging to catch noisy audio and traffic sounds more effectively.

---

## 🔧 Improvements Made

### 1. **More Comprehensive Detection (8 Checks)**
- **Check 3a**: High-frequency noise (centroid > 5500Hz + ZCR > 0.12)
- **Check 3b**: Very high centroid alone (>7000Hz)
- **Check 3c**: Excessive high-frequency content (rolloff > 9000Hz + centroid > 4500Hz)
- **Check 3d**: Very high rolloff alone (>11000Hz)
- **Check 3e**: Noise/static (high ZCR + low variation)
- **Check 3f**: Flat spectral content (low centroid variation)
- **Check 3g**: Low spectral contrast (new feature)
- **Check 3h**: High ZCR + high centroid combination

### 2. **Lowered Thresholds for Better Detection**
- **High-frequency threshold**: 6000Hz → 5500Hz (catches more traffic)
- **ZCR threshold**: 0.15 → 0.12 (catches more noise)
- **Rolloff threshold**: 10000Hz → 9000Hz (catches more high-pitch sounds)
- **SNR threshold**: 5.0dB → 3.0dB (less strict, only rejects extremely noisy)

### 3. **Added Spectral Contrast Analysis**
- New feature: Spectral contrast (difference between peaks and valleys)
- Music has higher contrast, noise has lower contrast
- Helps distinguish music from noise

### 4. **Added Debug Logging**
- Logs all spectral features: Centroid, ZCR, Rolloff, SNR
- Shows which check triggered rejection
- Helps identify why audio is accepted/rejected

### 5. **Added Max Value Checks**
- Now checks both average AND maximum values
- Catches brief high-pitch sounds (sirens, alarms)
- More robust detection

---

## 📊 Current Thresholds

| Check | Threshold | Purpose |
|-------|-----------|---------|
| SNR | 3.0 dB | Extremely noisy audio |
| Centroid (avg) | 5500 Hz | Traffic/mechanical sounds |
| Centroid (max) | 7000 Hz | Very high-frequency sounds |
| ZCR (avg) | 0.12 | Noise detection |
| ZCR (max) | 0.13 | Strong noise indicator |
| Rolloff (avg) | 9000 Hz | High-pitch sounds |
| Rolloff (max) | 11000 Hz | Very high-pitch (sirens) |
| Spectral Contrast | 2.0 | Musical structure |

---

## 🔍 Debug Output

When processing audio, you'll now see console output like:

```
Audio Quality Metrics - Centroid: 5234.5Hz (max: 6789.2Hz), ZCR: 0.089 (max: 0.134), Rolloff: 7845.3Hz (max: 9234.1Hz), SNR: 12.3dB
```

If rejected, you'll see:
```
REJECTED: High-frequency noise detected (centroid: 6234.5Hz, ZCR: 0.145)
```

---

## 🧪 Testing & Debugging

### To see what's happening:

1. **Check backend console logs** when uploading audio
2. Look for "Audio Quality Metrics" line - shows detected values
3. If rejected, look for "REJECTED:" line - shows which check failed
4. Compare values with thresholds above

### Example Scenarios:

**Traffic Sound:**
```
Centroid: 6500Hz, ZCR: 0.14
→ Should trigger Check 3a or 3b
```

**High-Pitch Siren:**
```
Rolloff: 10500Hz, Centroid: 6000Hz
→ Should trigger Check 3c or 3d
```

**Noisy Audio:**
```
SNR: 2.5dB, RMS: 0.02
→ Should trigger Check 2
```

**Music (should pass):**
```
Centroid: 3000Hz, ZCR: 0.08, Rolloff: 5000Hz, SNR: 15dB
→ Should pass all checks
```

---

## ⚙️ Adjusting Thresholds

If detection is too strict or too lenient, adjust these values in `check_audio_quality()`:

```python
# Line 154: SNR threshold
SNR_THRESHOLD = 3.0  # Lower = more strict

# Line 206: High-frequency threshold
HIGH_FREQ_THRESHOLD = 5500  # Lower = catches more traffic

# Line 207: ZCR threshold
HIGH_ZCR_THRESHOLD = 0.12  # Lower = catches more noise

# Line 214: Very high centroid
if avg_centroid > 7000:  # Lower = more strict

# Line 220: Rolloff threshold
if avg_rolloff > 9000:  # Lower = catches more high-pitch

# Line 225: Max rolloff
if max_rolloff > 11000:  # Lower = more strict
```

---

## 🐛 Troubleshooting

### If traffic sounds still get through:

1. **Check console logs** - what values are detected?
2. **Lower thresholds** - reduce HIGH_FREQ_THRESHOLD to 5000 or lower
3. **Check if SNR is high** - traffic might have good SNR but wrong spectral features

### If valid music is being rejected:

1. **Check console logs** - which check is triggering?
2. **Raise thresholds** - increase the threshold that's causing rejection
3. **Check if it's actually noisy** - maybe the music has background noise

### If nothing is being rejected:

1. **Check if function is being called** - look for "Checking audio quality" in logs
2. **Check detected values** - compare with thresholds
3. **Lower thresholds** - make detection more strict

---

## 📝 Next Steps

1. **Test with actual traffic sounds** - upload and check console logs
2. **Adjust thresholds** based on detected values
3. **Fine-tune** individual checks if needed
4. **Monitor** rejection rates to ensure balance

The debug logging will help you see exactly what's happening with each audio file!

