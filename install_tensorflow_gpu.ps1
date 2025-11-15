# TensorFlow GPU Installation Script
# This script installs TensorFlow with GPU support in your virtual environment

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "TensorFlow GPU Installation" -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Green
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "[2/4] Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install TensorFlow (includes GPU support automatically)
Write-Host "[3/4] Installing TensorFlow with GPU support..." -ForegroundColor Green
python -m pip install --upgrade tensorflow

# Verify installation
Write-Host "[4/4] Verifying GPU detection..." -ForegroundColor Green
Write-Host ""
python -c @"
import tensorflow as tf
print('=' * 70)
print('TensorFlow Version:', tf.__version__)
print('Built with CUDA:', tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs Detected: {len(gpus)}')
if gpus:
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu.name}')
    print('\n✓ SUCCESS: GPU is ready for use!')
else:
    print('\n⚠ WARNING: No GPU detected.')
    print('  If you have an NVIDIA GPU, you may need to:')
    print('  1. Install NVIDIA drivers')
    print('  2. Install CUDA Toolkit (11.8 or 12.x)')
    print('  3. Install cuDNN')
    print('\n  Or install: pip install tensorflow[and-cuda]')
print('=' * 70)
"@

Write-Host ""
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "You can now run: python train_model.py" -ForegroundColor Cyan
