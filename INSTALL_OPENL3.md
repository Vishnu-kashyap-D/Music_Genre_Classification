# Installing OpenL3 on Python 3.12

OpenL3 has compatibility issues with Python 3.12 due to the deprecated `imp` module. Here are your options:

## Option 1: Use Python 3.10 or 3.11 (Recommended if you need OpenL3)

If you need OpenL3 features, use Python 3.10 or 3.11:

```bash
# Create a new conda environment with Python 3.11
conda create -n music_genre python=3.11
conda activate music_genre

# Then install requirements
pip install -r requirements.txt
```

## Option 2: Install OpenL3 Manually (Python 3.12 workaround)

Try installing from a pre-built wheel or use an alternative method:

```bash
# Try installing from conda-forge (if available)
conda install -c conda-forge openl3

# Or try installing an older version
pip install openl3==0.4.1

# Or install from source with a patch
git clone https://github.com/marl/openl3.git
cd openl3
# Manually fix the setup.py to remove 'imp' import
pip install .
```

## Option 3: Use Mel Features Only (Current Setup)

**The web app works perfectly with mel features only!** OpenL3 is only needed if:
- Your model was specifically trained with `openl3` feature type
- You want to use OpenL3 embeddings for inference

If your model uses `mel` features (which is the default), you don't need OpenL3 at all.

## Check Your Model's Feature Type

You can check what feature type your model uses:

```python
import torch
payload = torch.load("torch_models/parallel_genre_classifier_torch.pt", map_location='cpu')
feature_type = payload.get("meta", {}).get("feature_type", "mel")
print(f"Model uses: {feature_type} features")
```

If it says `mel`, you don't need OpenL3!

