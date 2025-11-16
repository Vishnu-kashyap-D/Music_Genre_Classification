# 🚀 Run Commands - Quick Reference

## Backend (Flask API)

### Using Python 3.10 Environment (Recommended)

```powershell
# Activate environment
conda activate music_genre

# Run backend
python app.py
```

**Backend URL:** `http://localhost:5000`

### Using Python 3.12 (mel features only)

```powershell
python app.py
```

---

## Frontend (React App)

### First Time Setup

```powershell
cd frontend
npm install
```

### Run Frontend

```powershell
cd frontend
npm run dev
```

**Frontend URL:** `http://localhost:3000`

---

## Running Both Services

### Windows PowerShell

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

### Linux/Mac

**Terminal 1 (Backend):**
```bash
conda activate music_genre
python app.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```

---

## Verify Everything is Running

1. **Backend Health Check:**
   - Open: `http://localhost:5000/health`
   - Should return: `{"status": "healthy", "model_loaded": true}`

2. **Frontend:**
   - Open: `http://localhost:3000`
   - Should show the Music Genre Classifier interface

---

## Troubleshooting

**Backend not starting?**
- Check if port 5000 is in use
- Verify model exists: `torch_models/parallel_genre_classifier_torch.pt`
- Check Python version: `python --version`

**Frontend not starting?**
- Check if port 3000 is in use
- Verify Node.js version: `node --version` (should be 18+)
- Try: `cd frontend && rm -rf node_modules && npm install`

**Connection issues?**
- Ensure backend is running before starting frontend
- Check CORS is enabled (flask-cors installed)
- Verify both services are on correct ports


