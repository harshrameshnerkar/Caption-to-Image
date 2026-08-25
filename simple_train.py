"""
Alternative training script for Image Captioning that works without Hugging Face datasets
Uses simple_dataset.py for data loading and is compatible with Python 3.14
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import (
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup
)
from simple_dataset import create_simple_dataloaders, inspect_dataset
import matplotlib.pyplot as plt
from datetime import datetime
import json
from tqdm import tqdm

class SimpleImageCaptioningTrainer:
    """
    Simplified image captioning trainer that works without Hugging Face datasets
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model components
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.optimizer = None
        self.scheduler = None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def setup_model(self):
        """Initialize the Vision Encoder-Decoder model"""
        print("Setting up model...")
        
        try:
            # Load pre-trained model
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.config['model_name']
            )
            
            # Load image processor and tokenizer
            self.image_processor = ViTImageProcessor.from_pretrained(
                self.config['vision_model']
            )
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                self.config['text_model']
            )
            
            # Set special tokens
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
            
            # Move model to device
            self.model.to(self.device)
            
            print(f"✅ Model loaded: {self.config['model_name']}")
            print(f"✅ Image processor: {self.config['vision_model']}")
            print(f"✅ Tokenizer: {self.config['text_model']}")
            
        except Exception as e:
            print(f"❌ Error setting up model: {e}")
            raise
    
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data...")
        
        try:
            # First inspect the dataset
            inspect_dataset(
                self.config['captions_file'], 
                self.config['images_dir'],
                num_samples=3
            )
            
            # Create dataloaders
            self.train_dataloader, self.val_dataloader = create_simple_dataloaders(
                captions_file=self.config['captions_file'],
                images_dir=self.config['images_dir'],
                batch_size=self.config['batch_size'],
                val_split=self.config['val_split'],
                max_samples=self.config.get('max_samples', None),
                image_processor=self.image_processor
            )
            
            print(f"✅ Data loaders created successfully")
            
        except Exception as e:
            print(f"❌ Error setting up data: {e}")
            raise
    
    def setup_training(self):
        """Setup optimizer and scheduler"""
        print("Setting up training components...")
        
        # Calculate total steps
        total_steps = len(self.train_dataloader) * self.config['num_epochs']
        warmup_steps = int(0.1 * total_steps)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"✅ Optimizer: AdamW (lr={self.config['learning_rate']})")
        print(f"✅ Scheduler: Linear with warmup ({warmup_steps}/{total_steps})")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move batch to device
                images = batch['image'].to(self.device)
                captions = batch['caption']
                
                # Tokenize captions
                tokenized = self.tokenizer(
                    captions,
                    padding=True,
                    truncation=True,
                    max_length=self.config['max_caption_length'],
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=images,
                    labels=tokenized.input_ids
                )
                
                loss = outputs.loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Break early if testing with small dataset
                if self.config.get('max_samples') and batch_idx >= 5:
                    break
                    
            except Exception as e:
                print(f"❌ Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    images = batch['image'].to(self.device)
                    captions = batch['caption']
                    
                    # Tokenize captions
                    tokenized = self.tokenizer(
                        captions,
                        padding=True,
                        truncation=True,
                        max_length=self.config['max_caption_length'],
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        pixel_values=images,
                        labels=tokenized.input_ids
                    )
                    
                    total_loss += outputs.loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"❌ Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def generate_sample_captions(self, num_samples=3):
        """Generate sample captions to check training progress"""
        self.model.eval()
        
        print(f"\n📝 Sample Generated Captions:")
        
        with torch.no_grad():
            batch = next(iter(self.val_dataloader))
            images = batch['image'][:num_samples].to(self.device)
            true_captions = batch['caption'][:num_samples]
            
            # Generate captions
            generated_ids = self.model.generate(
                pixel_values=images,
                max_length=50,
                num_beams=3,
                do_sample=True,
                temperature=0.7
            )
            
            generated_captions = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            for i, (true_cap, gen_cap) in enumerate(zip(true_captions, generated_captions)):
                print(f"{i+1}. True: {true_cap[:60]}...")
                print(f"   Generated: {gen_cap[:60]}...")
                print()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best checkpoint: {best_path}")
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        if len(self.train_losses) == 0:
            return
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.config['checkpoint_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Training curves saved: {plot_path}")
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting training for {self.config['num_epochs']} epochs...")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{self.config['num_epochs']} ===")
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"📊 Train Loss: {train_loss:.4f}")
            print(f"📊 Val Loss: {val_loss:.4f}")
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"🎉 New best validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Generate sample captions
            if (epoch + 1) % 2 == 0:  # Every 2 epochs
                self.generate_sample_captions()
        
        print(f"\n✅ Training completed!")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves()

def main():
    """Main training function"""
    
    # Training configuration
    config = {
        # Model settings
        'model_name': 'nlpconnect/vit-gpt2-image-captioning',
        'vision_model': 'google/vit-base-patch16-224-in21k',
        'text_model': 'gpt2',
        
        # Data settings
        'captions_file': 'Dataset/captions.txt',
        'images_dir': 'Dataset/Images',
        'max_caption_length': 128,
        
        # Training settings
        'batch_size': 8,  # Reduced for compatibility
        'num_epochs': 3,  # Reduced for testing
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'val_split': 0.1,
        
        # Optional: limit samples for testing
        'max_samples': 100,  # Remove this line for full training
        
        # Output settings
        'checkpoint_dir': 'checkpoints',
    }
    
    # Print configuration
    print("=== Image Captioning Training ===")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    try:
        # Initialize trainer
        trainer = SimpleImageCaptioningTrainer(config)
        
        # Setup components
        trainer.setup_model()
        trainer.setup_data()
        trainer.setup_training()
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()