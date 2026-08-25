"""
Test script to check if PyTorch is available and working
"""
import sys
import importlib
import io

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def check_package(package_name):
    """Check if a package is available"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def test_pytorch_installation():
    """Test PyTorch installation and functionality"""
    print("Python version:", sys.version)
    print("\nTesting package availability:")
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("PIL", "Pillow"),
        ("flask", "Flask"),
        ("flask_cors", "Flask-CORS"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy")
    ]
    
    available_packages = []
    missing_packages = []
    
    for package, name in packages:
        if check_package(package):
            print(f"✅ {name} - Available")
            available_packages.append(name)
        else:
            print(f"❌ {name} - Missing")
            missing_packages.append(name)
    
    print(f"\nSummary:")
    print(f"Available packages: {len(available_packages)}")
    print(f"Missing packages: {len(missing_packages)}")
    
    if check_package("torch"):
        try:
            import torch
            print(f"\n🎉 PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            
            # Test basic tensor operations
            x = torch.randn(3, 3)
            print(f"Test tensor created successfully: {x.shape}")
            return True
        except Exception as e:
            print(f"❌ PyTorch import failed: {e}")
            return False
    else:
        print("\n⚠️ PyTorch not available - cannot run training")
        return False

if __name__ == "__main__":
    test_pytorch_installation()