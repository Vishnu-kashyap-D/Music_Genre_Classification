# 🚀 Quick Setup Guide

## Running the Application

### Step 1: Start Backend (Terminal 1)

**Using Python 3.10 (Recommended):**
```powershell
# Activate conda environment
conda activate music_genre

# Install dependencies (if not already done)
pip install -r requirements.txt

# Start Flask server
python app.py
```

**Or using Python 3.12:**
```powershell
pip install -r requirements.txt
python app.py
```

✅ Backend running on `http://localhost:5000`

### Step 2: Start Frontend (Terminal 2)

**Open a NEW terminal window:**
```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

✅ Frontend running on `http://localhost:3000`

### Step 3: Open Application

Open your browser and go to: **`http://localhost:3000`**

## ✅ Verify Setup

1. Check backend: Open `http://localhost:5000/health` in browser
   - Should return: `{"status": "healthy", "model_loaded": true}`

2. Check frontend: Open `http://localhost:3000` in browser
   - Should show the Music Genre Classifier interface

## 🎯 Test the Application

1. **Upload an audio file:**
   - Drag & drop an MP3/WAV file onto the upload zone
   - Or click to browse and select a file
   - Click "Analyze Genre"

2. **Or record live:**
   - Click "Start Recording"
   - Speak/sing for up to 10 seconds
   - Click "Stop Recording"
   - Click "Analyze Genre"

3. **View results:**
   - See genre predictions with confidence scores
   - Explore the radar chart and confidence bars
   - Play the waveform with timeline heatmap
   - View the spectrogram
   - Check audio quality (SNR meter)

## 📝 Notes

- Ensure the model file exists: `torch_models/parallel_genre_classifier_torch.pt`
- First request may take longer as the model loads
- Audio files should be at least 3 seconds long
- Maximum file size: 50MB

## 🐛 Troubleshooting

**Backend not starting?**
- Check if port 5000 is available
- Verify Python version (3.10+)
- Ensure all dependencies are installed

**Frontend not connecting?**
- Verify backend is running on port 5000
- Check browser console for errors
- Ensure CORS is enabled (flask-cors installed)

**Model not loading?**
- Check model file path in `app.py`
- Verify model file exists and is not corrupted
- Check console for error messages

