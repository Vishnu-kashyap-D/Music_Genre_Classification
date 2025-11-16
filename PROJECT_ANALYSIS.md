# Project Analysis: Music Genre Classification System

## 📋 Overview

This is a **Music Genre Classification** project that uses Deep Learning (Parallel CNN architecture) to classify music genres. The project consists of a **Flask backend** and a **React frontend** web application, along with comprehensive training and evaluation scripts.

---

## 🎯 What Has Been Done

### 1. **Core ML Model** ✅
- **Parallel CNN Architecture**: Multi-branch CNN that processes mel-spectrogram slices or OpenL3 embeddings
- **Multi-label Classification**: Supports multiple genres per audio file (sigmoid outputs with BCE loss)
- **Feature Types**: 
  - Mel-spectrogram features (default)
  - OpenL3 embeddings (optional, requires TensorFlow)
- **Advanced Features**:
  - SpecAugment for data augmentation
  - Residual + Squeeze-Excitation blocks
  - Attention pooling
  - CUDA acceleration with automatic mixed precision (AMP)
  - Deterministic seeding for reproducibility

### 2. **Training Pipeline** ✅
- **Script**: `train_parallel_cnn.py`
- **Features**:
  - Configurable windowing (default: 15s windows split into 3x5s slices)
  - Caching system for preprocessed features
  - TensorBoard logging
  - Model checkpointing with metadata
  - Train/validation/test split
  - Multi-label training with threshold-based evaluation

### 3. **Evaluation & Inference** ✅
- **Single-file inference**: `evaluate_parallel_model.py`
  - Top-3 genre reporting
  - Clip controls (`--clip-start`, `--clip-duration`, `--max-windows`)
  - GPU/CPU selection
- **Custom dataset evaluation**: `evaluate_custom_parallel.py`
  - CSV-based evaluation
  - Multi-label ground-truth parsing
  - Per-genre accuracy metrics
- **Legacy scripts**: Support for older TensorFlow and single-branch PyTorch models

### 4. **Backend API (Flask)** ✅
- **File**: `app.py`
- **Endpoints**:
  - `POST /predict`: Main prediction endpoint
  - `GET /health`: Health check
- **Features**:
  - Audio file upload (MP3, WAV, FLAC, M4A, OGG)
  - 3-second segment processing (up to 10 segments, 30 seconds max)
  - Per-segment genre predictions
  - Timeline analysis
  - SNR (Signal-to-Noise Ratio) calculation
  - Mel-spectrogram image generation (Base64 encoded)
  - Global confidence scores
  - Automatic model loading with fallback to CPU
  - CORS enabled for frontend

### 5. **Frontend Web Application (React)** ✅
- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS
- **Key Libraries**:
  - React Router for navigation
  - Framer Motion for animations
  - Chart.js for visualizations
  - WaveSurfer.js for audio waveform
  - React Hot Toast for notifications

#### **Pages**:
1. **Home** (`/`): Main landing page with audio upload
2. **Analysis** (`/analysis`): Shows loading and results
3. **History** (`/history`): Displays analysis history (protected route)
4. **Login** (`/login`): User authentication
5. **Signup** (`/signup`): User registration
6. **Profile** (`/profile`): User profile management (protected route)

#### **Components**:
- `AudioInput.jsx`: Drag & drop file upload with visual feedback
- `ResultsDashboard.jsx`: Main results display
- `RadarChart.jsx`: Genre similarity visualization
- `ConfidenceBars.jsx`: Color-coded confidence bars
- `WaveformPlayer.jsx`: Interactive audio player with timeline
- `SpectrogramView.jsx`: Mel-spectrogram visualization
- `SNRMeter.jsx`: Signal-to-noise ratio gauge
- `PDFGenerator.jsx`: Generate PDF reports
- `ShareButtons.jsx`: Social sharing (WhatsApp, Twitter, Instagram)
- `Loader.jsx`: Music-themed loading animation
- `Navbar.jsx`: Navigation with theme toggle
- `Sidebar.jsx`: History sidebar
- `Login.jsx` / `Signup.jsx`: Authentication forms
- `Profile.jsx`: User profile
- `ProtectedRoute.jsx`: Route protection

#### **Context Providers**:
- `ThemeContext.jsx`: Dark/light mode management
- `AuthContext.jsx`: User authentication state

#### **Features**:
- ✅ Dark/Light mode toggle (persisted in localStorage)
- ✅ Drag & drop file upload with animations
- ✅ Live microphone recording
- ✅ Real-time waveform visualization
- ✅ Genre predictions with confidence scores
- ✅ Timeline analysis (per-segment predictions)
- ✅ Spectrogram visualization
- ✅ SNR meter
- ✅ PDF report generation
- ✅ Social sharing
- ✅ Analysis history (localStorage)
- ✅ Responsive design
- ✅ Smooth animations

### 6. **Data Management** ✅
- **Dataset**: GTZAN dataset support
- **Preprocessing**: `preprocess_data.py` for legacy pipeline
- **Data caching**: Efficient caching system in `torch_cache/`
- **Upload folder**: `uploads/` for temporary file storage

### 7. **Documentation** ✅
- Comprehensive README files
- Setup guides
- Feature documentation
- Quick start guides
- Installation instructions

---

## 🏗️ Backend Structure

### **Main Files**:
```
backend/
├── app.py                          # Flask API server
├── train_parallel_cnn.py          # Main training script
├── evaluate_parallel_model.py     # Single-file inference
├── evaluate_custom_parallel.py    # CSV-based evaluation
├── train_model_torch.py           # Legacy single-branch training
├── predict_genre_torch.py         # Legacy inference
├── requirements.txt               # Python dependencies
└── torch_models/                  # Trained model checkpoints
    └── parallel_genre_classifier_torch.pt
```

### **Backend Architecture**:

1. **Model Architecture** (`train_parallel_cnn.py`):
   - `ParallelCNN`: Main model class
   - `MelSliceEncoder`: Processes mel-spectrogram slices
   - `EmbeddingSliceEncoder`: Processes OpenL3 embeddings
   - Multi-branch architecture with attention pooling
   - Supports shared or separate backbones per slice

2. **API Server** (`app.py`):
   - Flask application with CORS
   - Global model loading (lazy initialization)
   - Audio processing pipeline:
     - File validation
     - Audio loading (librosa)
     - Segment processing (3-second segments)
     - Model inference
     - Spectrogram generation
     - SNR calculation
   - Response formatting with timeline and metrics

3. **Dependencies**:
   - PyTorch 2.6.0 (CUDA 12.4)
   - librosa (audio processing)
   - Flask & Flask-CORS (API)
   - OpenL3 (optional, for embeddings)
   - matplotlib (spectrogram generation)
   - soundfile (audio I/O)

---

## 🎨 Frontend Structure

### **Directory Structure**:
```
frontend/
├── src/
│   ├── components/          # Reusable components
│   │   ├── AudioInput.jsx
│   │   ├── ResultsDashboard.jsx
│   │   ├── RadarChart.jsx
│   │   ├── ConfidenceBars.jsx
│   │   ├── WaveformPlayer.jsx
│   │   ├── SpectrogramView.jsx
│   │   ├── SNRMeter.jsx
│   │   ├── PDFGenerator.jsx
│   │   ├── ShareButtons.jsx
│   │   ├── Loader.jsx
│   │   ├── Navbar.jsx
│   │   ├── Sidebar.jsx
│   │   ├── Login.jsx
│   │   ├── Signup.jsx
│   │   ├── Profile.jsx
│   │   └── ProtectedRoute.jsx
│   ├── pages/               # Page components
│   │   ├── Home.jsx
│   │   ├── Analysis.jsx
│   │   └── History.jsx
│   ├── context/             # React contexts
│   │   ├── ThemeContext.jsx
│   │   └── AuthContext.jsx
│   ├── config/              # Configuration
│   │   └── api.js           # API endpoints
│   ├── utils/               # Utility functions
│   │   ├── historyStorage.js
│   │   └── themeClasses.js
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── package.json
├── vite.config.js
└── tailwind.config.js
```

### **Frontend Architecture**:

1. **Routing**:
   - React Router v6
   - Protected routes for authenticated pages
   - Navigation with active state indicators

2. **State Management**:
   - React Context API for theme and auth
   - LocalStorage for history and theme persistence
   - Component-level state for UI interactions

3. **API Integration**:
   - Axios/Fetch for backend communication
   - Configurable API base URL (environment variables)
   - Error handling with toast notifications

4. **Styling**:
   - Tailwind CSS for utility-first styling
   - Dark/light theme with CSS variables
   - Responsive design (mobile-first)
   - Framer Motion for animations

5. **Key Features**:
   - File upload with drag & drop
   - Audio recording (Web Audio API)
   - Real-time visualizations (Chart.js)
   - PDF generation (jsPDF + html2canvas)
   - Social sharing integration

---

## 📊 Data Flow

### **Training Flow**:
1. Load audio files from dataset directory
2. Extract features (mel-spectrogram or OpenL3)
3. Create windows and slices
4. Cache preprocessed features
5. Train model with multi-label BCE loss
6. Save checkpoint with metadata

### **Inference Flow (Web App)**:
1. User uploads audio file (frontend)
2. File sent to `/predict` endpoint (backend)
3. Backend loads audio, processes into segments
4. Model inference on each segment
5. Generate spectrogram image
6. Calculate SNR and metrics
7. Return JSON response with predictions
8. Frontend displays results with visualizations

---

## 🔧 Configuration

### **Backend Config** (`app.py`):
- Model path: `torch_models/parallel_genre_classifier_torch.pt`
- Upload folder: `uploads/`
- Max file size: 50MB
- Max analysis duration: 30 seconds
- Supported formats: MP3, WAV, FLAC, M4A, OGG

### **Frontend Config** (`src/config/api.js`):
- API base URL: `http://127.0.0.1:5000` (configurable via env)
- History limit: 5 analyses
- Theme: Dark mode default

### **Model Config** (`train_parallel_cnn.py`):
- Window duration: 15 seconds
- Slice duration: 5 seconds
- Window stride: 5 seconds
- Sample rate: 22050 Hz
- Mel bins: 128
- N_FFT: 2048
- Hop length: 512

---

## 🚀 Running the Project

### **Backend**:
```powershell
# Activate environment
conda activate music_genre  # or .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start server
python app.py
# Runs on http://localhost:5000
```

### **Frontend**:
```powershell
cd frontend

# Install dependencies (first time)
npm install

# Start dev server
npm run dev
# Runs on http://localhost:3000
```

---

## 📈 Project Status

### ✅ **Completed**:
- Parallel CNN model architecture
- Training pipeline with multi-label support
- Evaluation scripts
- Flask backend API
- React frontend with full UI
- Dark/light theme
- Authentication system (demo)
- History management
- PDF report generation
- Social sharing
- Spectrogram visualization
- Timeline analysis

### 🔄 **Potential Improvements**:
- Precision/recall/F1 reporting in evaluation
- ONNX/TorchScript export for deployment
- Saliency map visualization
- Real backend authentication (database)
- Cloud storage for history
- Batch processing API
- Model versioning
- API rate limiting
- Docker containerization

---

## 📝 Notes

- **Model**: Requires trained checkpoint at `torch_models/parallel_genre_classifier_torch.pt`
- **Authentication**: Currently demo-based (accepts any credentials)
- **History**: Stored in browser localStorage (not persistent across devices)
- **OpenL3**: Optional dependency, requires Python 3.10+ and TensorFlow
- **GPU**: Optional but recommended for faster inference
- **Dataset**: GTZAN dataset used for training (10 genres)

---

## 🎓 Technologies Used

### **Backend**:
- Python 3.10+
- PyTorch 2.6.0
- Flask 3.0+
- librosa
- OpenL3 (optional)
- NumPy, SciPy
- scikit-learn

### **Frontend**:
- React 18
- Vite
- Tailwind CSS
- Framer Motion
- Chart.js
- WaveSurfer.js
- React Router
- React Hot Toast

---

*Last Updated: Based on current codebase analysis*

