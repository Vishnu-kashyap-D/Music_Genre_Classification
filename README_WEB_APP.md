# Music Genre Classification Web App

Professional web application for music genre classification using Deep Learning.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (for OpenL3 support) or Python 3.12 (mel features only)
- Node.js 18+
- npm or yarn
- Trained model: `torch_models/parallel_genre_classifier_torch.pt`

### Backend Setup & Run

**Option 1: Using Python 3.10 (Recommended - supports both mel and OpenL3)**
```powershell
# Activate conda environment
conda activate music_genre

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start Flask server
python app.py
```

**Option 2: Using Python 3.12 (mel features only)**
```powershell
# Install dependencies
pip install -r requirements.txt

# Start Flask server
python app.py
```

✅ Backend will run on `http://localhost:5000`

**Verify backend is running:**
- Open `http://localhost:5000/health` in browser
- Should return: `{"status": "healthy", "model_loaded": true}`

### Frontend Setup & Run

**In a NEW terminal window:**

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

✅ Frontend will run on `http://localhost:3000`

**Open in browser:** `http://localhost:3000`

### Running Both Services

**Terminal 1 (Backend):**
```powershell
conda activate music_genre
python app.py
```

**Terminal 2 (Frontend):**
```powershell
cd frontend
npm run dev
```

**Then open:** `http://localhost:3000` in your browser

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn
- CUDA-capable GPU (optional, for faster inference)
- Trained model: `torch_models/parallel_genre_classifier_torch.pt`

## 🎯 Features

### Backend (`app.py`)
- ✅ Audio file upload (MP3, WAV, FLAC, M4A)
- ✅ 3-second segment processing (10 segments)
- ✅ Genre prediction per segment
- ✅ SNR (Signal-to-Noise Ratio) calculation
- ✅ Mel-spectrogram image generation (Base64)
- ✅ Timeline analysis with per-segment predictions
- ✅ Global confidence scores

### Frontend
- ✅ **Dark/Light Mode Toggle** - Premium theme with smooth transitions
- ✅ **Drag & Drop Upload** - Animated upload zone with hover effects
- ✅ **Live Recording** - Microphone recording with real-time waveform visualization
- ✅ **Analysis Loader** - Music-themed loading animation (spinning vinyl + equalizer)
- ✅ **Results Dashboard:**
  - Genre Radar Chart (circular similarity map)
  - Confidence Bars (color-coded: Green >75%, Yellow 40-75%, Red <40%)
  - Professional Waveform Player with timeline heatmap
  - Spectrogram view (switchable tabs)
  - SNR Meter (visual gauge)
- ✅ **History Sidebar** - Last 5 analyses saved to localStorage
- ✅ **Social Sharing:**
  - WhatsApp sharing
  - Instagram Story image download
  - Twitter sharing
- ✅ **PDF Report Generator** - Complete analysis report with charts and metrics

## 📁 Project Structure

```
project-1/
├── app.py                          # Flask backend
├── requirements.txt                # Python dependencies
├── torch_models/
│   └── parallel_genre_classifier_torch.pt  # Trained model
├── frontend/
│   ├── src/
│   │   ├── components/            # React components
│   │   ├── context/               # Theme context
│   │   ├── App.jsx                # Main app component
│   │   └── main.jsx               # Entry point
│   ├── package.json
│   └── vite.config.js
└── uploads/                        # Temporary upload folder (auto-created)
```

## 🔧 Configuration

### Backend
- Model path: `torch_models/parallel_genre_classifier_torch.pt`
- Upload folder: `uploads/` (auto-created)
- Max file size: 50MB
- Supported formats: MP3, WAV, FLAC, M4A, OGG

### Frontend
- API endpoint: `http://localhost:5000/predict`
- History limit: 5 analyses
- Theme: Dark mode by default (saved to localStorage)

## 📊 API Endpoints

### `POST /predict`
Upload audio file for genre classification.

**Request:**
- `Content-Type: multipart/form-data`
- Body: `audio` (file)

**Response:**
```json
{
  "global_confidence": {
    "jazz": 0.85,
    "rock": 0.10,
    ...
  },
  "timeline": [
    {
      "start": 0,
      "end": 3,
      "top_genre": "jazz",
      "confidence": 0.85,
      "all_probs": {...}
    },
    ...
  ],
  "spectrogram_image": "base64_string...",
  "metrics": {
    "snr": 12.5,
    "duration": 30.0,
    "num_segments": 10
  },
  "filename": "song.mp3"
}
```

### `GET /health`
Health check endpoint.

## 🎨 UI/UX Highlights

- **Premium Dark Theme** - Modern, professional design
- **Smooth Animations** - Framer Motion for fluid interactions
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Accessibility** - Keyboard navigation and ARIA labels
- **Error Handling** - User-friendly error messages with toast notifications
- **Loading States** - Beautiful loading animations

## 🐛 Troubleshooting

### Backend Issues

1. **Model not found:**
   - Ensure `parallel_genre_classifier_torch.pt` exists in `torch_models/`
   - Check model path in `app.py`

2. **CUDA errors:**
   - Model will fall back to CPU automatically
   - Check GPU drivers if using CUDA

3. **Audio processing errors:**
   - Ensure audio file is valid and not corrupted
   - Check file format is supported
   - Verify audio duration is at least 3 seconds

### Frontend Issues

1. **CORS errors:**
   - Ensure Flask-CORS is installed
   - Check backend is running on port 5000

2. **API connection failed:**
   - Verify backend is running
   - Check `http://localhost:5000/health`

3. **Build errors:**
   - Clear `node_modules` and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version (18+ required)

## 📝 Notes

- The model processes audio in 3-second segments
- Each segment is analyzed independently
- Global confidence is the average across all segments
- Timeline shows per-segment predictions with genre colors
- History is stored in browser localStorage (cleared on browser data clear)

## 🚀 Deployment

### Backend (Production)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (Production)
```bash
cd frontend
npm run build
# Serve dist/ folder with nginx or similar
```

## 📄 License

See main project README for license information.

## 👤 Author

Vishnu-kashyap-D

