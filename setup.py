#!/usr/bin/env python3
"""
Quick setup and test script for the Image Caption Generator
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header():
    """Print a welcome header"""
    print("=" * 60)
    print("🖼️  AI Image Caption Generator - Quick Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_dataset():
    """Check if dataset is available"""
    print("\n📁 Checking dataset...")
    
    dataset_dir = Path("Dataset")
    images_dir = dataset_dir / "Images"
    captions_file = dataset_dir / "captions.txt"
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found")
        return False
    
    if not images_dir.exists():
        print("❌ Images directory not found")
        return False
    
    if not captions_file.exists():
        print("❌ Captions file not found")
        return False
    
    # Count images and captions
    image_count = len([f for f in images_dir.iterdir() 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    try:
        import pandas as pd
        df = pd.read_csv(captions_file)
        caption_count = len(df)
        unique_images = df['image'].nunique()
    except:
        caption_count = 0
        unique_images = 0
    
    print(f"✅ Dataset found:")
    print(f"   📸 {image_count} images")
    print(f"   📝 {caption_count} captions")
    print(f"   🎯 {unique_images} unique images")
    
    return image_count > 0 and caption_count > 0

def check_gpu():
    """Check GPU availability"""
    print("\n🔧 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU available: {gpu_name}")
            print(f"   💾 {gpu_memory:.1f} GB VRAM")
            return True
        else:
            print("⚠️  No GPU available, will use CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("\n🧪 Running quick test...")
    
    try:
        # Test imports
        import torch
        import transformers
        from PIL import Image
        import flask
        import flask_cors
        
        print("✅ Core libraries imported successfully")
        
        # Test dataset loading
        from dataset import get_dataset_stats
        stats = get_dataset_stats("Dataset/captions.txt")
        print(f"✅ Dataset loading works - {stats['total_captions']} captions")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_sample_commands():
    """Create sample command files for easy execution"""
    print("\n📝 Creating sample command files...")
    
    # Training command
    train_cmd = """@echo off
echo Starting model training...
python train.py --epochs 5 --batch_size 8 --learning_rate 5e-5
echo Training completed!
pause
"""
    
    # Inference command
    inference_cmd = """@echo off
echo Running inference on sample image...
python inference.py --model_path models/best_model.pt --image_path Dataset/Images --style description
echo Inference completed!
pause
"""
    
    # Web app command
    app_cmd = """@echo off
echo Starting Flask web application...
python app.py
pause
"""
    
    try:
        with open("train_model.bat", "w") as f:
            f.write(train_cmd)
        
        with open("run_inference.bat", "w") as f:
            f.write(inference_cmd)
        
        with open("start_app.bat", "w") as f:
            f.write(app_cmd)
        
        print("✅ Command files created:")
        print("   📄 train_model.bat - Start training")
        print("   📄 run_inference.bat - Run inference")
        print("   📄 start_app.bat - Launch web app")
        
    except Exception as e:
        print(f"❌ Failed to create command files: {e}")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🚀 Setup Complete! Next Steps:")
    print("=" * 60)
    print()
    print("1. 📚 Train the model:")
    print("   python train.py")
    print("   or double-click: train_model.bat")
    print()
    print("2. 🌐 Launch web interface:")
    print("   python app.py") 
    print("   or double-click: start_app.bat")
    print()
    print("3. 💻 Run command-line inference:")
    print("   python inference.py --model_path models/best_model.pt --image_path your_image.jpg")
    print()
    print("4. 📖 Read the documentation:")
    print("   Check README.md for detailed instructions")
    print()
    print("💡 Tips:")
    print("   - Start with a small number of epochs (5-10) for initial testing")
    print("   - Use the web interface for interactive experimentation")
    print("   - Check the models/ directory for saved checkpoints")
    print()

def main():
    """Main setup function"""
    print_header()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing anyway, some features may not work")
    
    # Check dataset
    if not check_dataset():
        print("⚠️  Dataset issues detected, please check Dataset/ directory")
    
    # Check GPU
    check_gpu()
    
    # Run tests
    if not run_quick_test():
        print("⚠️  Some tests failed, but you can still try running the project")
    
    # Create convenience files
    create_sample_commands()
    
    # Show next steps
    print_next_steps()

if __name__ == "__main__":
    main()