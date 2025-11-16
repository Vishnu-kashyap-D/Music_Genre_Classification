# 🚀 Quick Start Guide

## ✅ Setup Complete!

Both **mel** and **OpenL3** features are now installed in the `music_genre` Python 3.10 environment.

## Starting the Application

### Backend (Flask)

**Option 1: Use the startup script**
```powershell
.\START_BACKEND.ps1
```

**Option 2: Manual activation**
```powershell
conda activate music_genre
python app.py
```

The backend will start on `http://localhost:5000`

### Frontend (React)

Open a new terminal and run:
```powershell
cd frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:3000`

## Verify Installation

Check that both features are available:
```powershell
conda activate music_genre
python -c "import openl3; import librosa; print('✅ OpenL3:', openl3.__version__); print('✅ Librosa:', librosa.__version__)"
```

## Features Available

✅ **Mel Features** - Fast mel-spectrogram processing  
✅ **OpenL3 Features** - Advanced audio embeddings  
✅ **Automatic Detection** - Backend detects feature type from model checkpoint  
✅ **Fallback Support** - Works with both feature types seamlessly

## Model Feature Type

Your model automatically uses the correct feature type:
- If model was trained with `--feature-type mel` → uses mel features
- If model was trained with `--feature-type openl3` → uses OpenL3 features

The backend detects this from the model checkpoint metadata.

## Next Steps

1. **Start backend:** `conda activate music_genre && python app.py`
2. **Start frontend:** `cd frontend && npm install && npm run dev`
3. **Open browser:** `http://localhost:3000`
4. **Upload audio** and enjoy! 🎵

