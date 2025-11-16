#!/bin/bash
# Bash script to set up Python 3.10 environment for Music Genre Classifier
# This allows both mel and OpenL3 features to work

echo "Setting up Python 3.10 environment for Music Genre Classifier..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo -e "\nFound conda! Creating Python 3.10 environment..."
    
    ENV_NAME="music_genre"
    
    # Check if environment already exists
    if conda env list | grep -q "$ENV_NAME"; then
        echo "Environment '$ENV_NAME' already exists. Activating..."
        conda activate $ENV_NAME
    else
        echo "Creating new conda environment '$ENV_NAME' with Python 3.10..."
        conda create -n $ENV_NAME python=3.10 -y
        
        echo -e "\nActivating environment..."
        conda activate $ENV_NAME
    fi
    
    echo -e "\nInstalling dependencies..."
    pip install -r requirements.txt
    
    echo -e "\n✅ Setup complete!"
    echo -e "\nTo activate this environment in the future, run:"
    echo "  conda activate music_genre"
    echo -e "\nThen start the backend with:"
    echo "  python app.py"
    
else
    echo -e "\n❌ Conda not found. Please install Anaconda or Miniconda first."
    echo -e "\nAlternative: Install Python 3.10 manually and create a virtual environment:"
    echo "  1. Install Python 3.10 from https://www.python.org/downloads/"
    echo "  2. Create venv: python3.10 -m venv venv"
    echo "  3. Activate: source venv/bin/activate"
    echo "  4. Install: pip install -r requirements.txt"
fi

