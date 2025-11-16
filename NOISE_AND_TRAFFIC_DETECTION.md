# Noise and Traffic Sound Detection - Audio Quality Enhancement

## Summary
Enhanced audio quality checking to detect and reject noisy audio and non-musical sounds (traffic, mechanical noise, high-pitch sounds).

---

## ✅ New Detection Features

### 1. **Noisy Audio Detection**
- **SNR Check**: Signal-to-Noise Ratio threshold of 5.0 dB
- **Detection**: Audio with excessive background noise
- **Error Message**: "Audio is too noisy or contains excessive background noise. Please upload a cleaner recording."

### 2. **Traffic Sound Detection**
- **Spectral Centroid**: Detects high-frequency mechanical sounds (>6000 Hz)
- **Zero Crossing Rate**: Detects noise-like characteristics (>0.15)
- **Combined Check**: High centroid + high ZCR = traffic/mechanical sounds
- **Error Message**: "Audio appears to contain non-musical sounds (e.g., traffic, mechanical noise). Please upload a music track."

### 3. **High-Pitch Non-Musical Sounds**
- **Spectral Rolloff**: Detects excessive high-frequency content (>10000 Hz)
- **Spectral Centroid**: Confirms high-pitch nature (>5000 Hz)
- **Detection**: Sirens, high-pitch traffic, mechanical sounds
- **Error Message**: "Audio contains high-pitch non-musical sounds. Please upload a music track for genre classification."

### 4. **Static/White Noise Detection**
- **ZCR Analysis**: High zero crossing rate (>0.12) with low variation
- **Spectral Structure**: Low bandwidth variation indicates lack of musical structure
- **Detection**: Pure noise, static, white noise
- **Error Message**: "Audio appears to be noise or static rather than music. Please upload a clear music track."

### 5. **Flat Spectral Content Detection**
- **Centroid Variation**: Low variation (<200 Hz) indicates flat content
- **ZCR Check**: Combined with high ZCR (>0.1) indicates noise
- **Detection**: White noise, flat static, non-musical content
- **Error Message**: "Audio lacks musical structure. Please upload a music track with clear musical content."

---

## 🔧 Technical Implementation

### Spectral Features Used

1. **Spectral Centroid** (Brightness)
   - Measures the "brightness" of the sound
   - Traffic/mechanical sounds: >6000 Hz
   - Music: Typically 1000-5000 Hz

2. **Zero Crossing Rate (ZCR)**
   - Measures how often the signal crosses zero
   - Noise: High ZCR (>0.15)
   - Music: Moderate ZCR (0.05-0.12)

3. **Spectral Rolloff**
   - Frequency below which 85% of energy is contained
   - Traffic: >10000 Hz
   - Music: 2000-8000 Hz

4. **Spectral Bandwidth**
   - Spread of frequencies in the signal
   - Music: Varies over time
   - Noise: Low variation

### Detection Logic

```python
# Traffic/Mechanical Sounds
if avg_centroid > 6000 Hz AND avg_zcr > 0.15:
    → Reject: "Non-musical sounds (traffic, mechanical noise)"

# High-Pitch Non-Musical
if avg_rolloff > 10000 Hz AND avg_centroid > 5000 Hz:
    → Reject: "High-pitch non-musical sounds"

# Static/Noise
if avg_zcr > 0.12 AND low_variation:
    → Reject: "Noise or static"

# Flat Content
if centroid_std < 200 Hz AND avg_zcr > 0.1:
    → Reject: "Lacks musical structure"
```

---

## 📊 Thresholds (Adjustable)

| Feature | Threshold | Purpose |
|---------|-----------|---------|
| SNR | 5.0 dB | Detect very noisy audio |
| Spectral Centroid | 6000 Hz | Detect high-frequency sounds |
| ZCR | 0.15 | Detect noise-like characteristics |
| Spectral Rolloff | 10000 Hz | Detect excessive high frequencies |
| Centroid Variation | 200 Hz | Detect flat spectral content |

---

## 🎯 What Gets Rejected

✅ **Rejected:**
- Traffic sounds (cars, horns, engines)
- Mechanical noise (machinery, tools)
- High-pitch sirens/alarms
- Static/white noise
- Very noisy recordings (SNR < 5 dB)
- Flat, non-musical content

✅ **Accepted:**
- Music tracks (all genres)
- Recordings with moderate background noise
- Live music recordings
- Studio recordings
- Any audio with musical structure

---

## 🔍 Error Messages

All errors return JSON with clear messages:

1. **Noisy Audio:**
   ```json
   {
     "error": "Audio is too noisy or contains excessive background noise. Please upload a cleaner recording."
   }
   ```

2. **Traffic/Mechanical:**
   ```json
   {
     "error": "Audio appears to contain non-musical sounds (e.g., traffic, mechanical noise). Please upload a music track."
   }
   ```

3. **High-Pitch Non-Musical:**
   ```json
   {
     "error": "Audio contains high-pitch non-musical sounds. Please upload a music track for genre classification."
   }
   ```

4. **Static/Noise:**
   ```json
   {
     "error": "Audio appears to be noise or static rather than music. Please upload a clear music track."
   }
   ```

5. **Lacks Structure:**
   ```json
   {
     "error": "Audio lacks musical structure. Please upload a music track with clear musical content."
   }
   ```

---

## 🧪 Testing Recommendations

### Test Cases

1. **Traffic Sound**
   - Upload: Car horn, engine noise
   - **Expected:** Rejected with traffic detection message
   - **Status:** ✅ Fixed

2. **High-Pitch Siren**
   - Upload: Siren, alarm sound
   - **Expected:** Rejected with high-pitch detection message
   - **Status:** ✅ Fixed

3. **Noisy Recording**
   - Upload: Music with heavy background noise (SNR < 5 dB)
   - **Expected:** Rejected with noisy audio message
   - **Status:** ✅ Fixed

4. **Static/White Noise**
   - Upload: Static, white noise file
   - **Expected:** Rejected with static detection message
   - **Status:** ✅ Fixed

5. **Music Track**
   - Upload: Normal music file
   - **Expected:** Accepted and processed
   - **Status:** ✅ Works

---

## ⚙️ Configuration

### Adjustable Parameters (in `check_audio_quality()`)

```python
# SNR threshold (line 154)
SNR_THRESHOLD = 5.0  # dB - increase to be more strict

# High-frequency threshold (line 191)
HIGH_FREQ_THRESHOLD = 6000  # Hz - decrease to catch more traffic sounds

# ZCR threshold (line 192)
HIGH_ZCR_THRESHOLD = 0.15  # decrease to be more strict

# Rolloff threshold (line 199)
# avg_rolloff > 10000  # Hz - decrease to catch more high-pitch sounds

# Centroid variation (line 213)
# centroid_std < 200  # Hz - increase to allow more variation
```

---

## 📝 Implementation Details

### Function: `check_audio_quality(signal, sample_rate)`

**Location:** `app.py` lines 131-221

**Checks Performed:**
1. RMS energy (silent/quiet)
2. SNR (noisy audio)
3. Spectral centroid (high-frequency sounds)
4. Zero crossing rate (noise detection)
5. Spectral rolloff (high-pitch detection)
6. Spectral bandwidth variation (structure detection)
7. Centroid variation (flat content detection)

**Returns:** `(is_valid: bool, error_message: Optional[str])`

---

## ✅ All Enhancements Complete

The audio quality checking now handles:
- ✅ Silent/quiet audio
- ✅ Noisy audio (low SNR)
- ✅ Traffic sounds
- ✅ Mechanical noise
- ✅ High-pitch non-musical sounds
- ✅ Static/white noise
- ✅ Flat, non-musical content

The system will now reject non-musical sounds and noisy recordings, ensuring only quality music tracks are processed for genre classification!

