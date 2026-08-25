import torch
from transformers import VisionEncoderDecoderModel, AutoProcessor
from PIL import Image
import os
import argparse
from typing import List, Dict, Any
import json
import numpy as np
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class CaptionStyleGenerator:
    """
    Generates captions in different styles: short, description, story
    """
    
    STYLE_PROMPTS = {
        'short': "A brief caption: ",
        'description': "A detailed description: ",
        'story': "A creative story about this image: "
    }
    
    STYLE_PARAMS = {
        'short': {
            'max_length': 15,
            'min_length': 5,
            'num_beams': 3,
            'temperature': 0.7,
            'length_penalty': 1.0
        },
        'description': {
            'max_length': 100,
            'min_length': 30,
            'num_beams': 5,
            'temperature': 0.8,
            'length_penalty': 1.2
        },
        'story': {
            'max_length': 200,
            'min_length': 80,
            'num_beams': 4,
            'temperature': 1.0,
            'length_penalty': 0.8,
            'do_sample': True,
            'top_p': 0.9
        }
    }

class ImageCaptionInference:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the inference model
        
        Args:
            model_path (str): Path to the trained model
            device (str): Device to run inference on
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load model and processor
        self.load_model()
        
        # Initialize style generator
        self.style_generator = CaptionStyleGenerator()
        
        # Initialize evaluation metrics
        self.setup_metrics()
        
    def load_model(self):
        """Load the trained model and processor"""
        print(f"Loading model from {self.model_path}")
        
        try:
            # Try loading from saved model directory
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except:
            # Try loading from checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Load from checkpoint config
            model_name = checkpoint.get('config', {}).get('model_name', 'nlpconnect/vit-gpt2-image-captioning')
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        
    def setup_metrics(self):
        """Setup evaluation metrics"""
        self.bleu = load("bleu")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for the model
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def generate_caption(self, image_path: str, style: str = 'description') -> Dict[str, Any]:
        """
        Generate a caption for an image in the specified style
        
        Args:
            image_path (str): Path to the image
            style (str): Caption style ('short', 'description', 'story')
            
        Returns:
            dict: Generated caption and metadata
        """
        if style not in self.style_generator.STYLE_PARAMS:
            raise ValueError(f"Invalid style. Choose from: {list(self.style_generator.STYLE_PARAMS.keys())}")
        
        # Preprocess image
        pixel_values = self.preprocess_image(image_path)
        if pixel_values is None:
            return {'error': 'Failed to process image'}
        
        # Get generation parameters for the style
        gen_params = self.style_generator.STYLE_PARAMS[style].copy()
        
        start_time = time.time()
        
        with torch.no_grad():
            # Generate caption
            output_ids = self.model.generate(
                pixel_values,
                **gen_params,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                early_stopping=True
            )
            
            # Decode caption
            caption = self.processor.tokenizer.decode(
                output_ids[0], 
                skip_special_tokens=True
            )
        
        generation_time = time.time() - start_time
        
        # Post-process caption based on style
        caption = self.post_process_caption(caption, style)
        
        return {
            'caption': caption,
            'style': style,
            'generation_time': generation_time,
            'caption_length': len(caption.split())
        }
    
    def post_process_caption(self, caption: str, style: str) -> str:
        """
        Post-process the generated caption based on style
        
        Args:
            caption (str): Raw generated caption
            style (str): Caption style
            
        Returns:
            str: Post-processed caption
        """
        # Remove style prompt if present
        for prompt in self.style_generator.STYLE_PROMPTS.values():
            if caption.startswith(prompt):
                caption = caption[len(prompt):].strip()
        
        # Style-specific post-processing
        if style == 'short':
            # Keep only the first sentence for short captions
            sentences = caption.split('.')
            caption = sentences[0].strip()
            if not caption.endswith('.'):
                caption += '.'
                
        elif style == 'story':
            # Ensure story format
            if not caption.endswith('.'):
                caption += '.'
            # Add some creative elements if too simple
            if len(caption.split()) < 20:
                # This is a simple heuristic - in practice, you might want more sophisticated story enhancement
                pass
        
        return caption.strip()
    
    def calculate_accuracy_metrics(self, generated_caption: str, reference_captions: List[str]) -> Dict[str, float]:
        """
        Calculate various accuracy metrics comparing generated caption with references
        
        Args:
            generated_caption (str): Generated caption
            reference_captions (List[str]): List of reference captions
            
        Returns:
            dict: Dictionary of accuracy metrics
        """
        metrics = {}
        
        # BLEU Score
        reference_tokens = [ref.lower().split() for ref in reference_captions]
        candidate_tokens = generated_caption.lower().split()
        
        # BLEU-1 to BLEU-4
        for i in range(1, 5):
            bleu_score = sentence_bleu(reference_tokens, candidate_tokens, weights=tuple([1/i]*i + [0]*(4-i)))
            metrics[f'bleu_{i}'] = bleu_score
        
        # ROUGE Scores
        rouge_scores = []
        for ref in reference_captions:
            scores = self.rouge_scorer.score(ref, generated_caption)
            rouge_scores.append(scores)
        
        # Average ROUGE scores
        rouge_metrics = {}
        for metric_type in ['rouge1', 'rouge2', 'rougeL']:
            precision_scores = [score[metric_type].precision for score in rouge_scores]
            recall_scores = [score[metric_type].recall for score in rouge_scores]
            f1_scores = [score[metric_type].fmeasure for score in rouge_scores]
            
            rouge_metrics[f'{metric_type}_precision'] = np.mean(precision_scores)
            rouge_metrics[f'{metric_type}_recall'] = np.mean(recall_scores)
            rouge_metrics[f'{metric_type}_f1'] = np.mean(f1_scores)
        
        metrics.update(rouge_metrics)
        
        # METEOR Score (if available)
        try:
            meteor_scores = []
            for ref in reference_captions:
                score = meteor_score([ref.split()], candidate_tokens)
                meteor_scores.append(score)
            metrics['meteor'] = np.mean(meteor_scores)
        except:
            metrics['meteor'] = 0.0
        
        # Semantic similarity (simplified - could use BERT score in practice)
        # For now, use word overlap as a proxy
        gen_words = set(generated_caption.lower().split())
        ref_words = set()
        for ref in reference_captions:
            ref_words.update(ref.lower().split())
        
        if len(ref_words) > 0:
            word_overlap = len(gen_words.intersection(ref_words)) / len(ref_words)
            metrics['word_overlap'] = word_overlap
        else:
            metrics['word_overlap'] = 0.0
        
        # Overall accuracy score (weighted combination)
        overall_score = (
            metrics.get('bleu_4', 0) * 0.3 +
            metrics.get('rouge1_f1', 0) * 0.3 +
            metrics.get('rougeL_f1', 0) * 0.2 +
            metrics.get('meteor', 0) * 0.2
        )
        metrics['overall_accuracy'] = overall_score
        
        return metrics
    
    def batch_inference(self, image_paths: List[str], style: str = 'description') -> List[Dict[str, Any]]:
        """
        Run inference on multiple images
        
        Args:
            image_paths (List[str]): List of image paths
            style (str): Caption style
            
        Returns:
            List[Dict]: List of results for each image
        """
        results = []
        
        for img_path in image_paths:
            print(f"Processing: {os.path.basename(img_path)}")
            result = self.generate_caption(img_path, style)
            result['image_path'] = img_path
            results.append(result)
        
        return results
    
    def evaluate_on_dataset(self, csv_file: str, img_dir: str, num_samples: int = 100, style: str = 'description') -> Dict[str, float]:
        """
        Evaluate the model on a subset of the dataset
        
        Args:
            csv_file (str): Path to captions file
            img_dir (str): Path to images directory
            num_samples (int): Number of samples to evaluate
            style (str): Caption style
            
        Returns:
            dict: Average metrics across all samples
        """
        import pandas as pd
        from collections import defaultdict
        
        # Load dataset
        df = pd.read_csv(csv_file)
        
        # Group captions by image
        image_captions = defaultdict(list)
        for _, row in df.iterrows():
            image_captions[row['image']].append(row['caption'])
        
        # Sample images
        images = list(image_captions.keys())[:num_samples]
        
        all_metrics = []
        
        print(f"Evaluating on {len(images)} images with style: {style}")
        
        for i, img_name in enumerate(images):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(images)}")
            
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                continue
            
            # Generate caption
            result = self.generate_caption(img_path, style)
            
            if 'error' in result:
                continue
            
            # Calculate metrics
            reference_captions = image_captions[img_name]
            metrics = self.calculate_accuracy_metrics(result['caption'], reference_captions)
            all_metrics.append(metrics)
        
        # Average metrics
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics

def main():
    parser = argparse.ArgumentParser(description='Image Caption Inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--image_path', help='Path to single image for inference')
    parser.add_argument('--image_dir', help='Directory of images for batch inference')
    parser.add_argument('--style', choices=['short', 'description', 'story'], default='description', help='Caption style')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on dataset')
    parser.add_argument('--csv_file', default='Dataset/captions.txt', help='Path to captions file for evaluation')
    parser.add_argument('--img_dir', default='Dataset/Images', help='Path to images directory for evaluation')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples for evaluation')
    parser.add_argument('--output_file', help='Path to save results')
    
    args = parser.parse_args()
    
    # Initialize inference model
    inferencer = ImageCaptionInference(args.model_path)
    
    if args.evaluate:
        # Evaluate on dataset
        print("Running evaluation on dataset...")
        metrics = inferencer.evaluate_on_dataset(
            args.csv_file, 
            args.img_dir, 
            args.num_samples, 
            args.style
        )
        
        print(f"\nEvaluation Results (Style: {args.style}):")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    elif args.image_path:
        # Single image inference
        result = inferencer.generate_caption(args.image_path, args.style)
        
        print(f"\nImage: {args.image_path}")
        print(f"Style: {args.style}")
        print(f"Caption: {result['caption']}")
        print(f"Generation Time: {result['generation_time']:.3f}s")
        print(f"Caption Length: {result['caption_length']} words")
    
    elif args.image_dir:
        # Batch inference
        image_paths = []
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            image_paths.extend([
                os.path.join(args.image_dir, f) 
                for f in os.listdir(args.image_dir) 
                if f.lower().endswith(ext)
            ])
        
        results = inferencer.batch_inference(image_paths[:10], args.style)  # Limit to 10 for demo
        
        print(f"\nBatch Inference Results (Style: {args.style}):")
        print("-" * 50)
        for result in results:
            print(f"Image: {os.path.basename(result['image_path'])}")
            print(f"Caption: {result['caption']}")
            print()
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    else:
        print("Please specify --image_path, --image_dir, or --evaluate")

if __name__ == "__main__":
    main()