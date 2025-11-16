# PowerShell script to set up Python 3.10 environment for Music Genre Classifier
# This allows both mel and OpenL3 features to work

Write-Host "Setting up Python 3.10 environment for Music Genre Classifier..." -ForegroundColor Cyan

# Check if conda is available
$condaAvailable = Get-Command conda -ErrorAction SilentlyContinue

if ($condaAvailable) {
    Write-Host "`nFound conda! Creating Python 3.10 environment..." -ForegroundColor Green
    
    $envName = "music_genre"
    
    # Check if environment already exists
    $envExists = conda env list | Select-String $envName
    
    if ($envExists) {
        Write-Host "Environment '$envName' already exists. Activating..." -ForegroundColor Yellow
        conda activate $envName
    } else {
        Write-Host "Creating new conda environment '$envName' with Python 3.10..." -ForegroundColor Green
        conda create -n $envName python=3.10 -y
        
        Write-Host "`nActivating environment..." -ForegroundColor Green
        conda activate $envName
    }
    
    Write-Host "`nInstalling dependencies..." -ForegroundColor Green
    pip install -r requirements.txt
    
    Write-Host "`n✅ Setup complete!" -ForegroundColor Green
    Write-Host "`nTo activate this environment in the future, run:" -ForegroundColor Cyan
    Write-Host "  conda activate music_genre" -ForegroundColor White
    Write-Host "`nThen start the backend with:" -ForegroundColor Cyan
    Write-Host "  python app.py" -ForegroundColor White
    
} else {
    Write-Host "`n❌ Conda not found. Please install Anaconda or Miniconda first." -ForegroundColor Red
    Write-Host "`nAlternative: Install Python 3.10 manually and create a virtual environment:" -ForegroundColor Yellow
    Write-Host "  1. Download Python 3.10 from https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "  2. Create venv: python3.10 -m venv venv" -ForegroundColor White
    Write-Host "  3. Activate: .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "  4. Install: pip install -r requirements.txt" -ForegroundColor White
}

