# Music Genre Classification (Parallel CNN + OpenL3)

This repository now hosts a feature-rich multi-label genre classifier built around a parallel CNN
backbone. Key upgrades delivered so far:

- ✅ Multi-branch architecture that ingests mel-spectrogram slices and optional OpenL3 embeddings
	per window, fused via attention pooling for stronger predictions.
- ✅ Full multi-label training loop (sigmoid outputs) with CUDA acceleration and automatic mixed
	precision.
- ✅ Single-file inference with top-3 reporting plus precise clip controls (`--clip-start`,
	`--clip-duration`, `--max-windows`).
- ✅ Custom CSV evaluator for datasets outside GTZAN, including multi-label ground-truth parsing.
- ✅ Caching utilities, TensorBoard-friendly logging, and deterministic seeding.

The project still includes the legacy single-branch PyTorch/TensorFlow pipelines for comparison,
but the parallel CNN flow described below is the recommended path.

---

## 🚀 Quick Start - Clone and Run on Any System

### Prerequisites
- **Python 3.11+** (Python 3.10 recommended for full OpenL3 support)
- **Node.js 16+** and npm
- **Git**
- **NVIDIA GPU** (optional, but recommended for faster processing)
- **CUDA 12.4+** drivers (if using GPU)

### Step 1: Clone the Repository (pro1 branch)

```powershell
# Clone the pro1 branch (deployment-ready with pre-trained model)
git clone -b pro1 https://github.com/Vishnu-kashyap-D/Music_Genre_Classification.git
cd Music_Genre_Classification
```

### Step 2: Setup Python Backend

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows CMD:
.\.venv\Scripts\activate.bat

# Install Python dependencies (includes PyTorch with CUDA support)
pip install -r requirements.txt
```

### Step 3: Setup React Frontend

```powershell
# Open a new terminal and navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install
```

### Step 4: Run the Application

**Terminal 1 - Start Backend Server:**
```powershell
# From project root directory
.\.venv\Scripts\Activate.ps1
python app.py
```
✅ Backend will run on `http://127.0.0.1:5000`

**Terminal 2 - Start Frontend Server:**
```powershell
# From project root directory
cd frontend
npm run dev
```
✅ Frontend will run on `http://localhost:3000`

### Step 5: Access the Application

Open your browser and go to: **http://localhost:3000**

### 🎵 What's Included Out of the Box

- ✅ **Pre-trained Model**: `torch_models/parallel_genre_classifier_torch.pt` (56MB, 10 genres)
- ✅ **Complete Backend**: Flask API with CUDA/CPU support
- ✅ **Full Frontend**: React app with authentication, analysis, history tracking
- ✅ **No Training Needed**: Model is ready to classify audio immediately

### 📊 Supported Genres

The pre-trained model classifies 10 genres:
- Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

### 🔧 Troubleshooting

**Port Already in Use:**
```powershell
# Kill process on port 5000 (backend)
Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process -Force

# Kill process on port 3000 (frontend)
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process -Force
```

**Python Module Not Found:**
```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\Activate.ps1
# Reinstall dependencies
pip install -r requirements.txt
```

**Frontend Build Errors:**
```powershell
# Clear npm cache and reinstall
cd frontend
rm -r node_modules
npm install
```

---

## 0. Get the Code (Advanced)

```powershell
# grab the repository
git clone https://github.com/Vishnu-kashyap-D/Music_Genre_Classification.git
cd Music_Genre_Classification

# (optional) create an isolated environment right away
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Note:** Model checkpoints (`*.pt`, `*.joblib`, `genre_classifier.keras`) and raw datasets are
ignored by Git. If you want to reuse a trained model, copy the checkpoint into `torch_models/`
after cloning (or download it from the release you share).

---

## 1. Environment Setup

```powershell
# optional: create the venv first (python -m venv .venv)
V:\path\to\project> .\.venv\Scripts\Activate.ps1

# install all dependencies (PyTorch CUDA wheels + OpenL3 + librosa, etc.)
(.venv) V:\path\to\project> pip install -r requirements.txt
```

`requirements.txt` already pins the CUDA 12.4 PyTorch wheel via `--extra-index-url`, so the only
host prerequisite is an up-to-date NVIDIA driver. TensorFlow is pulled in automatically for
OpenL3 embedding extraction.

---

## 2. Data Preparation

1. **Download GTZAN** – you can unpack it under `archive/Data/` or use the helper script:

	 ```powershell
	 (.venv) V:\path\to\project> bash download_dataset.sh
	 ```

	 (Run inside WSL or Git Bash; otherwise download manually and place under `Data/`.)

2. **Optional preprocessing JSON** – legacy scripts rely on `data.json`:

	 ```powershell
	 (.venv) V:\path\to\project> python preprocess_data.py
	 ```

	 The parallel CNN pipeline streams WAV files directly, so `data.json` is only required if you
	 still use `train_model.py`/`predict_genre.py`.

---

## 3. Training the Parallel CNN

```powershell
(.venv) V:\path\to\project> python train_parallel_cnn.py ^
		--dataset Data/genres_original ^
		--feature-type openl3 ^
		--epochs 60 ^
		--batch-size 16 ^
		--save-path torch_models/parallel_genre_classifier_torch.pt
```

### Essential arguments

| Flag | Description |
| --- | --- |
| `--feature-type {mel,openl3}` | Primary modality. `openl3` automatically loads/configures the pre-trained embedder. |
| `--gpu-index / --cpu` | Select device manually. CUDA + AMP is on by default when a GPU is visible. |
| `--window-duration`, `--slice-duration`, `--window-stride` | Control temporal windowing (default 15s window split into three 5s slices, hopping every 5s). |
| `--lr`, `--weight-decay`, `--epochs`, `--batch-size` | Standard optimisation knobs. |
| `--openl3-*` | Override embedding dim, repr, hop size, etc., when `feature-type=openl3`. |

All randomness is seeded via `set_seed`, and checkpoints store both model weights and metadata
needed for inference (window sizes, feature type, OpenL3 config, class mapping).

---

## 4. Single-File Inference (Parallel CNN)

```powershell
(.venv) V:\path\to\project> python evaluate_parallel_model.py ^
		"C:\test_set\A. Cooper - 1,440 Minutes.mp3" ^
		--model .\torch_models\parallel_genre_classifier_torch.pt ^
		--clip-duration 5 ^
		--clip-start 30
```

What you get:

- Multi-label predictions above the 0.5 threshold (can be adjusted in code) plus top-3 scores.
- Ability to limit evaluation to a short excerpt (e.g., a 5-second slice starting 30 seconds into
	the track) or to cap the number of 15-second windows using `--max-windows`.
- GPU/CPU selection identical to training.

---

## 5. Evaluating a Custom Dataset

Prepare a CSV with columns `file,label`. Separate multiple ground-truth genres with `|` (case-insensitive).

```csv
file,label
C:\test_set\Liberty Beats - Aether's Dream.mp3,Pop|Rock
relative\path\to\sample.wav,Jazz
```

Run the evaluator (use `--root` when the CSV stores relative paths):

```powershell
(.venv) V:\path\to\project> python evaluate_custom_parallel.py ^
		.\Data\custom_eval.csv ^
		--model .\torch_models\parallel_genre_classifier_torch.pt ^
		--root "C:\test_set"
```

The script prints overall accuracy plus per-genre accuracy (using thresholded multi-label hits with
a top-1 fallback). Samples with unknown labels or missing files are skipped with informative
messages.

---

## 6. Legacy Scripts (Optional)

| Script | Purpose |
| --- | --- |
| `train_model_torch.py` / `predict_genre_torch.py` | Earlier single-branch CNN (mel-only) but still useful for quick baselines. |
| `train_model.py` / `predict_genre.py` | Original TensorFlow implementation backed by `data.json`. |
| `evaluate_pop_hiphop.py` | Specialised auxiliary classifier for disambiguating pop vs hip-hop false positives. |

---

## 7. Project Status & Next Steps

Recent milestones:

- Converted training/testing loops to true multi-label objectives with sigmoid outputs and BCE loss.
- Added SpecAugment, residual + squeeze-excitation blocks, attention pooling, and optional OpenL3
	embeddings to the architecture.
- Built GPU-friendly evaluation tooling plus CLI flags for clip slicing and deterministic window
	selection.
- Implemented CSV-driven custom evaluation (multi-label) and improved single-file inference with
	always-on top-3 reporting.

Potential follow-ups:

- Add precision/recall/F1 reporting in `evaluate_custom_parallel.py`.
- Export ONNX / TorchScript packages for lightweight deployment.
- Provide notebooks that visualise saliency maps for interpretability.

---

## 8. Running the Web Application

### Backend (Flask API)

**Using Python 3.10 environment (recommended for OpenL3 support):**
```powershell
# Activate conda environment
conda activate music_genre

# Or if using venv
.\.venv\Scripts\Activate.ps1

# Start Flask server
python app.py
```

**Or using Python 3.12 (mel features only):**
```powershell
python app.py
```

Backend runs on `http://localhost:5000`

### Frontend (React App)

**In a new terminal:**
```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Frontend runs on `http://localhost:3000`

### Quick Start (Both Services)

**Terminal 1 - Backend:**
```powershell
conda activate music_genre
python app.py
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm run dev
```

Then open `http://localhost:3000` in your browser.

---

## 9. Quick Reference Commands

```powershell
# train parallel CNN with mel features only
python train_parallel_cnn.py --feature-type mel --epochs 40

# run inference on a single MP3, limiting to first 5 seconds
python evaluate_parallel_model.py "path\to\song.mp3" --clip-duration 5

# evaluate custom CSV (ensure files exist or supply --root)
python evaluate_custom_parallel.py Data/custom_eval.csv --root "C:\test_set"
```

Feel free to open issues or PRs if you explore different datasets or add new downstream tasks!
>>>>>>> Stashed changes
