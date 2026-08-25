import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoProcessor
from collections import defaultdict

class FlickrDataset(Dataset):
    """
    Custom Dataset for Flickr8k image captioning data
    """
    def __init__(self, csv_file, img_dir, processor, max_length=50, transform=None):
        """
        Args:
            csv_file (str): Path to the captions.txt file
            img_dir (str): Path to directory containing images
            processor: HuggingFace processor for the model
            max_length (int): Maximum sequence length for captions
            transform: Optional image transformations
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length
        self.transform = transform
        
        # Group captions by image
        self.image_captions = defaultdict(list)
        for _, row in self.df.iterrows():
            self.image_captions[row['image']].append(row['caption'])
        
        # Get unique images
        self.images = list(self.image_captions.keys())
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a default black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Get random caption for this image during training
        captions = self.image_captions[img_name]
        caption = captions[torch.randint(0, len(captions), (1,)).item()]
        
        # Process image and caption
        if self.processor is not None:
            # Use processor for vision-encoder-decoder models
            encoding = self.processor(
                images=image,
                text=caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            
        else:
            # Manual processing if needed
            if self.transform:
                image = self.transform(image)
            encoding = {
                'pixel_values': image,
                'labels': caption
            }
            
        encoding['caption'] = caption
        encoding['image_path'] = img_path
        
        return encoding

def get_transforms():
    """
    Get image transforms for preprocessing
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(csv_file, img_dir, processor, batch_size=8, val_split=0.1, max_length=50):
    """
    Create train and validation dataloaders
    
    Args:
        csv_file (str): Path to captions.txt
        img_dir (str): Path to images directory
        processor: HuggingFace processor
        batch_size (int): Batch size for training
        val_split (float): Fraction of data to use for validation
        max_length (int): Maximum caption length
    
    Returns:
        tuple: (train_loader, val_loader, dataset_info)
    """
    # Load full dataset
    df = pd.read_csv(csv_file)
    
    # Get unique images
    unique_images = df['image'].unique()
    
    # Split images (not individual captions) for train/val
    num_val = int(len(unique_images) * val_split)
    
    # Shuffle images
    import random
    random.shuffle(unique_images)
    
    val_images = set(unique_images[:num_val])
    train_images = set(unique_images[num_val:])
    
    # Split dataframe
    train_df = df[df['image'].isin(train_images)]
    val_df = df[df['image'].isin(val_images)]
    
    # Save splits to temporary files
    train_csv = csv_file.replace('.txt', '_train.txt')
    val_csv = csv_file.replace('.txt', '_val.txt')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Create datasets
    train_dataset = FlickrDataset(
        csv_file=train_csv,
        img_dir=img_dir,
        processor=processor,
        max_length=max_length
    )
    
    val_dataset = FlickrDataset(
        csv_file=val_csv,
        img_dir=img_dir,
        processor=processor,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    dataset_info = {
        'num_train_images': len(train_images),
        'num_val_images': len(val_images),
        'num_train_captions': len(train_df),
        'num_val_captions': len(val_df),
        'vocab_size': len(df),  # This would need proper tokenizer vocab size
    }
    
    return train_loader, val_loader, dataset_info

def collate_fn(batch):
    """
    Custom collate function for batching
    """
    # Extract different components
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Handle labels (captions) - need to pad them
    if 'labels' in batch[0]:
        labels = [item['labels'] for item in batch]
        # This would need proper padding based on tokenizer
    
    return {
        'pixel_values': pixel_values,
        'labels': labels if 'labels' in batch[0] else None,
        'captions': [item['caption'] for item in batch],
        'image_paths': [item['image_path'] for item in batch]
    }

def get_dataset_stats(csv_file):
    """
    Get statistics about the dataset
    """
    df = pd.read_csv(csv_file)
    
    stats = {
        'total_captions': len(df),
        'unique_images': df['image'].nunique(),
        'avg_captions_per_image': len(df) / df['image'].nunique(),
        'caption_lengths': df['caption'].str.len().describe(),
        'vocab_size_estimate': len(set(' '.join(df['caption']).split()))
    }
    
    return stats

if __name__ == "__main__":
    # Test the dataset
    csv_file = "Dataset/captions.txt"
    img_dir = "Dataset/Images"
    
    # Get dataset statistics
    stats = get_dataset_stats(csv_file)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")