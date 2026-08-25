# AI Image Caption Generator

A comprehensive end-to-end image captioning system that generates captions in three different styles: short descriptions, detailed descriptions, and creative stories. Built with PyTorch, Transformers, and Streamlit.

## Features

- **Three Caption Styles:**
  - 🏃 **Short**: Brief 8-10 word captions
  - 📖 **Description**: Detailed 80-100 word descriptions  
  - 📚 **Story**: Creative 2000-3000 word stories

- **Advanced Model Training:**
  - Based on Vision Encoder-Decoder architecture (ViT-GPT2)
  - Fine-tuned on Flickr8k dataset
  - Configurable training parameters
  - Automatic evaluation metrics

- **Interactive Web Interface:**
  - Upload images or use sample images
  - Real-time caption generation
  - Text-to-speech audio playback
  - Accuracy metrics visualization
  - Batch processing capabilities

- **Comprehensive Evaluation:**
  - BLEU, ROUGE, METEOR scores
  - Word overlap analysis
  - Visual accuracy reports
  - Dataset evaluation tools

## Project Structure

```
CAPTIONTOIMAGE/
├── Dataset/
│   ├── captions.txt          # Image-caption pairs
│   └── Images/               # Training images
├── models/                   # Saved model checkpoints
├── requirements.txt          # Python dependencies
├── dataset.py               # Dataset loading and preprocessing
├── train.py                 # Model training script
├── inference.py             # Caption generation and evaluation
├── app.py                   # Streamlit web interface
└── README.md               # This file
```

## Quick Start

### 1. Installation

Clone the repository and install dependencies:

```powershell
git clone <your-repo-url>
cd CAPTIONTOIMAGE
pip install -r requirements.txt
```

### 2. Dataset Preparation

The project uses the Flickr8k dataset format. Your `Dataset/captions.txt` should have the format:
```csv
image,caption
image1.jpg,A description of the image
image1.jpg,Another description of the same image
...
```

### 3. Training the Model

Start training with default parameters:

```powershell
python train.py
```

Or customize training parameters:

```powershell
python train.py --epochs 15 --batch_size 16 --learning_rate 1e-4 --model_name "nlpconnect/vit-gpt2-image-captioning"
```

Training options:
- `--csv_file`: Path to captions file (default: `Dataset/captions.txt`)
- `--img_dir`: Path to images directory (default: `Dataset/Images`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 8)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_length`: Maximum caption length (default: 50)
- `--save_dir`: Directory to save models (default: `models`)
- `--use_wandb`: Enable Weights & Biases logging

### 4. Running the Web Interface

Launch the Streamlit web application:

```powershell
streamlit run app.py
```

The interface will be available at `http://localhost:8501`

### 5. Command Line Inference

Generate captions from command line:

```powershell
# Single image
python inference.py --model_path models/best_model.pt --image_path path/to/image.jpg --style description

# Batch processing
python inference.py --model_path models/best_model.pt --image_dir path/to/images --style short

# Evaluation on dataset
python inference.py --model_path models/best_model.pt --evaluate --num_samples 100 --style story
```

## Usage Examples

### Training

```powershell
# Basic training
python train.py

# Training with custom parameters
python train.py --epochs 20 --batch_size 16 --learning_rate 1e-4

# Training with Weights & Biases logging
python train.py --use_wandb --project_name my-captioning-project
```

### Inference

```powershell
# Generate short caption
python inference.py --model_path models/best_model.pt --image_path test_image.jpg --style short

# Generate detailed description
python inference.py --model_path models/best_model.pt --image_path test_image.jpg --style description

# Generate creative story
python inference.py --model_path models/best_model.pt --image_path test_image.jpg --style story

# Evaluate model performance
python inference.py --model_path models/best_model.pt --evaluate --num_samples 200
```

### Web Interface Features

1. **Image Upload**: Drag and drop or browse to upload images
2. **Style Selection**: Choose between short, description, or story styles
3. **Real-time Generation**: Get captions in seconds
4. **Audio Playback**: Listen to generated captions
5. **Accuracy Metrics**: See detailed performance metrics
6. **Batch Processing**: Process multiple images at once
7. **Model Evaluation**: Comprehensive performance testing

## Model Architecture

The system uses a Vision Encoder-Decoder model:

- **Encoder**: Vision Transformer (ViT) for image feature extraction
- **Decoder**: GPT-2 for text generation
- **Fine-tuning**: Trained on domain-specific image-caption pairs

### Caption Styles

1. **Short Captions** (8-10 words):
   - Quick, concise descriptions
   - Perfect for accessibility or quick reference
   - Example: "A dog playing in the park"

2. **Detailed Descriptions** (80-100 words):
   - Comprehensive scene descriptions
   - Include objects, actions, and context
   - Example: "A golden retriever dog is running through a green park on a sunny day. The dog appears happy and energetic as it chases a red ball across the grass. In the background, there are trees and other park visitors enjoying the outdoor space."

3. **Creative Stories** (2000-3000 words):
   - Narrative interpretations of images
   - Include character development and plot
   - Creative and engaging storytelling

## Evaluation Metrics

The system provides comprehensive evaluation using:

- **BLEU Scores**: N-gram precision metrics (BLEU-1 to BLEU-4)
- **ROUGE Scores**: Recall-oriented metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- **METEOR**: Semantic similarity with stemming and synonyms
- **Word Overlap**: Simple vocabulary coverage metric
- **Overall Accuracy**: Weighted combination of all metrics

## Performance Tips

### For Training:
- Use GPU if available (automatic detection)
- Adjust batch size based on GPU memory
- Monitor validation loss to prevent overfitting
- Use learning rate scheduling for better convergence

### For Inference:
- Batch processing is more efficient for multiple images
- GPU acceleration significantly improves speed
- Different styles have different computational requirements

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   ```powershell
   # Reduce batch size
   python train.py --batch_size 4
   ```

2. **Model Loading Errors**:
   ```powershell
   # Check model path and file existence
   python inference.py --model_path models/best_model.pt --image_path test.jpg
   ```

3. **Dataset Format Issues**:
   - Ensure CSV has 'image' and 'caption' columns
   - Check image file paths are correct
   - Verify image formats are supported (jpg, png, etc.)

4. **Audio Not Working**:
   - Ensure you have an internet connection for gTTS
   - Check browser audio permissions

### Performance Optimization:

1. **Training Speed**:
   - Use larger batch sizes on powerful GPUs
   - Enable mixed precision training
   - Use multiple workers for data loading

2. **Inference Speed**:
   - Use batch processing for multiple images
   - Consider model quantization for deployment
   - Cache model loading in production

## Dependencies

Key dependencies (see requirements.txt for full list):

- `torch>=2.0.0`: Deep learning framework
- `transformers>=4.30.0`: Hugging Face models
- `streamlit>=1.25.0`: Web interface
- `PIL>=9.0.0`: Image processing
- `nltk>=3.8`: Natural language processing
- `rouge-score>=0.1.2`: Evaluation metrics
- `gtts>=2.3.0`: Text-to-speech
- `matplotlib>=3.5.0`: Visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flickr8k dataset for training data
- Hugging Face for transformer models
- Streamlit for the web interface
- OpenAI for GPT architecture inspiration

## Future Enhancements

- [ ] Support for more image formats
- [ ] Additional caption styles (technical, poetic, etc.)
- [ ] Multi-language caption generation
- [ ] Real-time video captioning
- [ ] Mobile app development
- [ ] Advanced story generation with character consistency
- [ ] Integration with social media platforms
- [ ] Custom model architecture experiments

---

For questions or support, please open an issue on the repository or contact the development team.