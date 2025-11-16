# Setting Up with Both Mel and OpenL3 Features

To use both **mel** and **OpenL3** features, you need Python 3.10 or 3.11 (OpenL3 doesn't work with Python 3.12).

## Quick Setup (Recommended)

### Option 1: Using Conda (Easiest)

**Windows (PowerShell):**
```powershell
.\setup_python310_env.ps1
```

**Linux/Mac (Bash):**
```bash
chmod +x setup_python310_env.sh
./setup_python310_env.sh
```

Or manually:
```bash
# Create Python 3.10 environment
conda create -n music_genre python=3.10 -y
conda activate music_genre

# Install all dependencies (including OpenL3)
pip install -r requirements.txt
```

### Option 2: Using Python 3.10 venv

1. **Download Python 3.10:**
   - Windows: https://www.python.org/downloads/release/python-31011/
   - Or use pyenv: `pyenv install 3.10.11`

2. **Create virtual environment:**
   ```bash
   # Windows
   python3.10 -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # Linux/Mac
   python3.10 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Verify Installation

Check that OpenL3 is installed:
```bash
python -c "import openl3; print('OpenL3 version:', openl3.__version__)"
```

## Using Both Features

The backend automatically supports both:

1. **Mel Features (Default):**
   - Fast processing
   - Works with models trained using `--feature-type mel`
   - No additional setup needed

2. **OpenL3 Features:**
   - More accurate embeddings
   - Works with models trained using `--feature-type openl3`
   - Requires OpenL3 installation (now available with Python 3.10)

## Model Feature Type

Your model's feature type is stored in the checkpoint. The backend automatically:
- Detects the feature type from model metadata
- Uses the appropriate processing (mel or openl3)
- Falls back to mel if OpenL3 is not available

## Start the Application

```bash
# Activate environment (if using conda)
conda activate music_genre

# Or activate venv
# Windows: .\venv\Scripts\Activate.ps1
# Linux/Mac: source venv/bin/activate

# Start backend
python app.py
```

## Troubleshooting

**OpenL3 still not installing?**
- Ensure you're using Python 3.10 or 3.11
- Check: `python --version`
- Try: `pip install --upgrade pip setuptools wheel`
- Then: `pip install openl3`

**Model uses openl3 but OpenL3 not available?**
- The backend will show a warning and fall back to mel features
- Install OpenL3 using the steps above to use openl3 features

