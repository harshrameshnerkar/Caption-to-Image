
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_DIR = PROJECT_ROOT / "Dataset"
IMAGES_DIR = DATASET_DIR / "Images"
CAPTIONS_FILE = DATASET_DIR / "captions.txt"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configurations
MODEL_CONFIGS = {
    "vit-gpt2": {
        "name": "nlpconnect/vit-gpt2-image-captioning",
        "type": "vision-encoder-decoder",
        "description": "ViT encoder + GPT2 decoder for image captioning"
    },
    "blip-base": {
        "name": "Salesforce/blip-image-captioning-base",
        "type": "blip",
        "description": "BLIP model for image captioning"
    },
    "blip-large": {
        "name": "Salesforce/blip-image-captioning-large", 
        "type": "blip",
        "description": "Large BLIP model for better performance"
    }
}

# Training configurations
TRAINING_CONFIG = {
    "default": {
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_length": 50,
        "val_split": 0.1,
        "save_every": 2,
        "warmup_ratio": 0.1
    },
    "quick": {
        "epochs": 5,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "max_length": 30,
        "val_split": 0.1,
        "save_every": 1,
        "warmup_ratio": 0.05
    },
    "intensive": {
        "epochs": 20,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.005,
        "max_length": 75,
        "val_split": 0.15,
        "save_every": 3,
        "warmup_ratio": 0.15
    }
}

# Caption style configurations
CAPTION_STYLES = {
    "short": {
        "max_length": 15,
        "min_length": 5,
        "num_beams": 3,
        "temperature": 0.7,
        "length_penalty": 1.0,
        "description": "Brief 8-10 word captions"
    },
    "description": {
        "max_length": 100,
        "min_length": 30,
        "num_beams": 5,
        "temperature": 0.8,
        "length_penalty": 1.2,
        "description": "Detailed 80-100 word descriptions"
    },
    "story": {
        "max_length": 200,
        "min_length": 80,
        "num_beams": 4,
        "temperature": 1.0,
        "length_penalty": 0.8,
        "do_sample": True,
        "top_p": 0.9,
        "description": "Creative 2000-3000 word stories"
    }
}

# Evaluation metrics configuration
EVALUATION_CONFIG = {
    "metrics": ["bleu", "rouge", "meteor", "word_overlap"],
    "bleu_weights": {
        "bleu_1": [1.0, 0, 0, 0],
        "bleu_2": [0.5, 0.5, 0, 0], 
        "bleu_3": [0.33, 0.33, 0.33, 0],
        "bleu_4": [0.25, 0.25, 0.25, 0.25]
    },
    "rouge_types": ["rouge1", "rouge2", "rougeL"],
    "overall_weights": {
        "bleu_4": 0.3,
        "rouge1_f1": 0.3,
        "rougeL_f1": 0.2,
        "meteor": 0.2
    }
}

# Streamlit app configuration
APP_CONFIG = {
    "title": "AI Image Caption Generator",
    "page_icon": "🖼️",
    "layout": "wide",
    "max_file_size": 10,  # MB
    "supported_formats": ["jpg", "jpeg", "png", "bmp"],
    "default_style": "description",
    "enable_audio": True,
    "max_batch_size": 50
}

# Device configuration
DEVICE_CONFIG = {
    "auto_detect": True,
    "fallback_device": "cpu",
    "memory_fraction": 0.8  # GPU memory fraction to use
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "image_captioning.log",
    "console": True
}

# Data preprocessing
PREPROCESSING_CONFIG = {
    "image_size": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "max_caption_length": 200,
    "min_caption_length": 3
}

# Ensure directories exist
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [MODELS_DIR, DATASET_DIR, IMAGES_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return True

# Validate configuration
def validate_config():
    """Validate that all necessary files and directories exist"""
    issues = []
    
    # Check if dataset directory exists
    if not DATASET_DIR.exists():
        issues.append(f"Dataset directory not found: {DATASET_DIR}")
    
    # Check if images directory exists
    if not IMAGES_DIR.exists():
        issues.append(f"Images directory not found: {IMAGES_DIR}")
    
    # Check if captions file exists
    if not CAPTIONS_FILE.exists():
        issues.append(f"Captions file not found: {CAPTIONS_FILE}")
    
    return issues

# Get configuration by name
def get_training_config(config_name="default"):
    """Get training configuration by name"""
    if config_name in TRAINING_CONFIG:
        return TRAINING_CONFIG[config_name].copy()
    else:
        print(f"Warning: Unknown config '{config_name}', using default")
        return TRAINING_CONFIG["default"].copy()

def get_model_config(model_name="vit-gpt2"):
    """Get model configuration by name"""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()
    else:
        print(f"Warning: Unknown model '{model_name}', using vit-gpt2")
        return MODEL_CONFIGS["vit-gpt2"].copy()

def get_style_config(style_name="description"):
    """Get caption style configuration by name"""
    if style_name in CAPTION_STYLES:
        return CAPTION_STYLES[style_name].copy()
    else:
        print(f"Warning: Unknown style '{style_name}', using description")
        return CAPTION_STYLES["description"].copy()

# Export commonly used configurations
DEFAULT_MODEL = "vit-gpt2"
DEFAULT_STYLE = "description"  
DEFAULT_TRAINING = "default"

if __name__ == "__main__":
    # Test configuration
    print("Image Captioning Project Configuration")
    print("=" * 40)
    
    # Create directories
    create_directories()
    print("✓ Directories created/verified")
    
    # Validate configuration
    issues = validate_config()
    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ Configuration validated successfully")
    
    # Print key configurations
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Dataset Directory: {DATASET_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    
    print(f"\nAvailable Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Available Styles: {list(CAPTION_STYLES.keys())}")
    print(f"Training Configs: {list(TRAINING_CONFIG.keys())}")