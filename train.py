import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    VisionEncoderDecoderModel, 
    AutoProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from dataset import create_dataloaders, get_dataset_stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import wandb
import os
import json
from datetime import datetime
import argparse

class ImageCaptioningTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and processor
        self.setup_model()
        
        # Setup data
        self.setup_data()
        
        # Setup training components
        self.setup_training()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def setup_model(self):
        """Initialize the vision-encoder-decoder model"""
        print(f"Loading model: {self.config['model_name']}")
        
        # Load model and processor
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config['model_name'])
        self.processor = AutoProcessor.from_pretrained(self.config['model_name'])
        
        # Configure model for captioning
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Set generation parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = self.config['max_length']
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        self.train_loader, self.val_loader, self.dataset_info = create_dataloaders(
            csv_file=self.config['csv_file'],
            img_dir=self.config['img_dir'],
            processor=self.processor,
            batch_size=self.config['batch_size'],
            val_split=self.config['val_split'],
            max_length=self.config['max_length']
        )
        
        print(f"Training images: {self.dataset_info['num_train_images']}")
        print(f"Validation images: {self.dataset_info['num_val_images']}")
        print(f"Training captions: {self.dataset_info['num_train_captions']}")
        
    def setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Calculate total steps
        total_steps = len(self.train_loader) * self.config['epochs']
        warmup_steps = int(0.1 * total_steps)
        
        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log to wandb
            if self.config['use_wandb']:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': current_lr,
                    'epoch': epoch + batch_idx / num_batches
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    pixel_values=pixel_values,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        print(f"Validation Loss: {avg_loss:.4f}")
        
        if self.config['use_wandb']:
            wandb.log({
                'val_loss': avg_loss,
                'epoch': epoch
            })
        
        return avg_loss
    
    def generate_sample_captions(self, num_samples=3):
        """Generate sample captions for visualization"""
        self.model.eval()
        samples = []
        
        with torch.no_grad():
            # Get a few validation samples
            val_iter = iter(self.val_loader)
            batch = next(val_iter)
            
            for i in range(min(num_samples, len(batch['pixel_values']))):
                pixel_values = batch['pixel_values'][i:i+1].to(self.device)
                
                # Generate caption
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.config['max_length'],
                    num_beams=4,
                    early_stopping=True
                )
                
                generated_caption = self.processor.tokenizer.decode(
                    generated_ids[0], 
                    skip_special_tokens=True
                )
                
                samples.append({
                    'image_path': batch['image_paths'][i],
                    'ground_truth': batch['captions'][i],
                    'generated': generated_caption
                })
        
        return samples
    
    def save_model(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")
        
        # Save model and processor separately for easy loading
        model_dir = os.path.join(self.config['save_dir'], f'model_epoch_{epoch}')
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.processor.save_pretrained(model_dir)
        
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(1, 3, 2)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # Sample captions
        plt.subplot(1, 3, 3)
        samples = self.generate_sample_captions(1)
        if samples:
            sample = samples[0]
            plt.text(0.1, 0.7, f"Ground Truth:\n{sample['ground_truth']}", 
                    fontsize=8, wrap=True, transform=plt.gca().transAxes)
            plt.text(0.1, 0.3, f"Generated:\n{sample['generated']}", 
                    fontsize=8, wrap=True, transform=plt.gca().transAxes)
            plt.title('Sample Caption')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['save_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if self.config['use_wandb']:
            wandb.log({"training_curves": wandb.Image(plot_path)})
        
        plt.show()
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Create save directory
        os.makedirs(self.config['save_dir'], exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.config['save_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Check if best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Save model
            if (epoch + 1) % self.config['save_every'] == 0 or is_best:
                self.save_model(epoch, val_loss, is_best)
            
            # Generate sample captions
            if (epoch + 1) % 5 == 0:
                samples = self.generate_sample_captions()
                print("\nSample Generated Captions:")
                for i, sample in enumerate(samples):
                    print(f"Sample {i+1}:")
                    print(f"  Ground Truth: {sample['ground_truth']}")
                    print(f"  Generated: {sample['generated']}")
                    print()
        
        # Final plots
        self.plot_training_curves()
        
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train Image Captioning Model')
    parser.add_argument('--csv_file', default='Dataset/captions.txt', help='Path to captions file')
    parser.add_argument('--img_dir', default='Dataset/Images', help='Path to images directory')
    parser.add_argument('--model_name', default='nlpconnect/vit-gpt2-image-captioning', help='Pre-trained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=50, help='Max caption length')
    parser.add_argument('--save_dir', default='models', help='Directory to save models')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--project_name', default='image-captioning', help='WandB project name')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'csv_file': args.csv_file,
        'img_dir': args.img_dir,
        'model_name': args.model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 0.01,
        'max_length': args.max_length,
        'val_split': 0.1,
        'save_dir': args.save_dir,
        'save_every': 2,
        'use_wandb': args.use_wandb,
        'project_name': args.project_name
    }
    
    # Initialize wandb if requested
    if config['use_wandb']:
        wandb.init(
            project=config['project_name'],
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Print dataset stats
    stats = get_dataset_stats(config['csv_file'])
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create trainer and start training
    trainer = ImageCaptioningTrainer(config)
    trainer.train()
    
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()