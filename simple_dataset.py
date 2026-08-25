"""
Alternative dataset loader that doesn't require pyarrow/datasets library
This works directly with CSV files and is compatible with Python 3.14
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from typing import Tuple, Optional, List

class SimpleFlickrDataset(Dataset):
    """
    Simple dataset loader for Flickr8k that doesn't require Hugging Face datasets
    Works directly with captions.txt file and images folder
    """
    
    def __init__(self, captions_file: str, images_dir: str, transform=None, max_samples: Optional[int] = None):
        """
        Initialize the dataset
        
        Args:
            captions_file: Path to captions.txt file
            images_dir: Path to images directory
            transform: Optional image transforms
            max_samples: Limit the number of samples (for testing)
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Load captions
        self.image_captions = self._load_captions(captions_file, max_samples)
        
        print(f"Loaded {len(self.image_captions)} image-caption pairs")
    
    def _load_captions(self, captions_file: str, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
        """Load captions from the captions.txt file"""
        image_captions = []
        
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if max_samples and len(image_captions) >= max_samples:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split by comma (assuming format: image_name,caption)
                    parts = line.split(',', 1)
                    if len(parts) >= 2:
                        image_name = parts[0].strip()
                        caption = parts[1].strip()
                        
                        # Check if image exists
                        image_path = os.path.join(self.images_dir, image_name)
                        if os.path.exists(image_path):
                            image_captions.append((image_name, caption))
                        elif line_num <= 10:  # Only warn for first few missing files
                            print(f"Warning: Image not found: {image_path}")
                    elif line_num <= 5:  # Only warn for first few malformed lines
                        print(f"Warning: Malformed line {line_num}: {line[:50]}...")
        
        except Exception as e:
            print(f"Error loading captions: {e}")
            return []
        
        return image_captions
    
    def __len__(self):
        return len(self.image_captions)
    
    def __getitem__(self, idx):
        image_name, caption = self.image_captions[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
        
        return {
            'image': image,
            'caption': caption,
            'image_name': image_name
        }

def create_simple_dataloaders(
    captions_file: str,
    images_dir: str,
    batch_size: int = 16,
    val_split: float = 0.1,
    max_samples: Optional[int] = None,
    image_processor=None
):
    """
    Create train and validation dataloaders without using Hugging Face datasets
    
    Args:
        captions_file: Path to captions.txt
        images_dir: Path to images directory  
        batch_size: Batch size for dataloaders
        val_split: Fraction of data for validation
        max_samples: Limit total samples (for testing)
        image_processor: Optional image processor from transformers
    
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    
    # Define transform
    transform = None
    if image_processor:
        # Use the processor from transformers
        def transform_fn(image):
            return image_processor(image, return_tensors="pt")['pixel_values'][0]
        transform = transform_fn
    else:
        # Simple transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create full dataset
    full_dataset = SimpleFlickrDataset(
        captions_file=captions_file,
        images_dir=images_dir,
        transform=transform,
        max_samples=max_samples
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    print(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_dataloader, val_dataloader

def inspect_dataset(captions_file: str, images_dir: str, num_samples: int = 5):
    """
    Inspect the dataset to show sample data
    """
    print("=== Dataset Inspection ===")
    
    # Check if files exist
    if not os.path.exists(captions_file):
        print(f"❌ Captions file not found: {captions_file}")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    print(f"✅ Found captions file: {captions_file}")
    print(f"✅ Found images directory: {images_dir}")
    
    # Count images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📁 Found {len(image_files)} image files")
    
    # Read sample captions
    print(f"\n📝 Sample captions:")
    try:
        with open(captions_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                line = line.strip()
                if line:
                    parts = line.split(',', 1)
                    if len(parts) >= 2:
                        image_name = parts[0].strip()
                        caption = parts[1].strip()
                        image_path = os.path.join(images_dir, image_name)
                        exists = "✅" if os.path.exists(image_path) else "❌"
                        print(f"{i+1}. {exists} {image_name}: {caption[:60]}...")
    except Exception as e:
        print(f"❌ Error reading captions: {e}")

if __name__ == "__main__":
    # Test the dataset loader
    captions_file = "Dataset/captions.txt"
    images_dir = "Dataset/Images"
    
    inspect_dataset(captions_file, images_dir)
    
    # Test creating a small dataset
    print("\n=== Testing Dataset Creation ===")
    try:
        train_loader, val_loader = create_simple_dataloaders(
            captions_file=captions_file,
            images_dir=images_dir,
            batch_size=2,
            max_samples=10  # Small test
        )
        
        # Test loading a batch
        batch = next(iter(train_loader))
        print(f"✅ Successfully loaded batch:")
        print(f"   - Image tensor shape: {batch['image'].shape}")
        print(f"   - Number of captions: {len(batch['caption'])}")
        print(f"   - Sample caption: {batch['caption'][0][:50]}...")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")