# Alternative Installation Guide for Python 3.14

Since you're running Python 3.14, PyTorch pre-built wheels are not available yet.
Here are your options:

## Option 1: Use CPU-only PyTorch (Recommended for now)
Try installing the nightly build which might support Python 3.14:

```powershell
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

## Option 2: Install from source (Advanced)
This will take much longer but should work:

```powershell
pip install --upgrade setuptools wheel
pip install torch --no-binary torch
```

## Option 3: Use Python 3.11 or 3.12 (Easiest)
Consider using a Python version that PyTorch officially supports:
- Python 3.11 (Recommended)
- Python 3.12 (Also supported)

You can install multiple Python versions and use virtual environments.

## Option 4: Try with minimal dependencies first
Let's try installing just the basic packages without PyTorch to test the setup:

```powershell
pip install transformers datasets Pillow streamlit pandas numpy scikit-learn
```

## Current Status
Your Python version (3.14) is newer than what PyTorch currently supports.
The project will work once PyTorch is available for Python 3.14 or you switch to Python 3.11/3.12.