"""
Unified AI Image Caption Generator
Single-file executable for patenting
Includes: Flask app, dataset loader, training logic
"""
import os
import base64
import io
import shutil
import tempfile
import traceback
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast, AutoProcessor, AutoTokenizer, get_linear_schedule_with_warmup
import random

# --- Dataset Loader ---
class FlickrDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor, max_length=50, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.processor = processor
        self.max_length = max_length
        self.transform = transform
        self.image_captions = {}
        for _, row in self.df.iterrows():
            self.image_captions.setdefault(row['image'], []).append(row['caption'])
        self.images = list(self.image_captions.keys())
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color='black')
        captions = self.image_captions[img_name]
        caption = random.choice(captions)
        if self.processor is not None:
            encoding = self.processor(images=image, text=caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
            encoding = {k: v.squeeze() for k, v in encoding.items()}
        else:
            encoding = {'pixel_values': image, 'labels': caption}
        encoding['caption'] = caption
        encoding['image_path'] = img_path
        return encoding

def get_dataset_stats(csv_file):
    df = pd.read_csv(csv_file)
    stats = {
        'total_captions': len(df),
        'unique_images': df['image'].nunique(),
        'avg_captions_per_image': len(df) / df['image'].nunique(),
        'caption_lengths': df['caption'].str.len().describe().to_dict(),
        'vocab_size_estimate': len(set(' '.join(df['caption']).split()))
    }
    return stats

# --- Training Logic (Class Skeleton) ---
class ImageCaptioningTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VisionEncoderDecoderModel.from_pretrained(config['model_name'])
        self.processor = AutoProcessor.from_pretrained(config['model_name'])
        self.model.to(self.device)
    # Training methods omitted for brevity (see train.py for full logic)

# --- Flask App ---
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

model = None
processor = None
tokenizer = None
model_loaded = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_ai_model():
    global model, processor, tokenizer, model_loaded
    try:
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model_loaded = True
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False

def enhance_caption_with_style(base_caption, style="balanced"):
    if style == "short":
        words = base_caption.lower().split()
        subjects, actions, locations, objects = [], [], [], []
        subject_words = ['person', 'man', 'woman', 'child', 'people', 'dog', 'cat', 'animal', 'bird']
        action_words = ['sitting', 'standing', 'walking', 'running', 'playing', 'eating', 'holding', 'wearing']
        location_words = ['park', 'street', 'room', 'kitchen', 'beach', 'forest', 'garden', 'outdoor', 'indoor']
        object_words = ['bench', 'car', 'tree', 'house', 'table', 'chair', 'flower', 'water', 'food', 'book']
        for word in words:
            for subject in subject_words:
                if subject in word and subject not in subjects:
                    subjects.append(subject)
            for action in action_words:
                if action in word and action not in actions:
                    actions.append(action)
            for location in location_words:
                if location in word and location not in locations:
                    locations.append(location)
            for obj in object_words:
                if obj in word and obj not in objects:
                    objects.append(obj)
        description_parts = []
        if subjects:
            description_parts.append(subjects[0])
        if actions:
            description_parts.append(actions[0])
        if objects:
            description_parts.extend(objects[:2])
        elif locations:
            description_parts.append(locations[0])
        remaining_words = [w for w in words if len(w) > 3 and w not in ['with', 'and', 'the', 'in', 'on', 'at']]
        for word in remaining_words:
            if len(description_parts) < 8 and word not in ' '.join(description_parts):
                description_parts.append(word)
        if len(description_parts) < 8:
            additional_words = [w for w in words if len(w) > 2 and w not in description_parts]
            description_parts.extend(additional_words[:8-len(description_parts)])
        final_description = ' '.join(description_parts[:10])
        if len(description_parts) >= 8:
            return final_description
        else:
            return f"{base_caption.split('.')[0][:60]}..."
    elif style == "balanced":
        words = base_caption.lower().split()
        subjects, settings, colors, actions = [], [], [], []
        subject_words = ['person', 'man', 'woman', 'child', 'dog', 'cat', 'animal', 'car', 'building', 'tree', 'flower']
        for word in words:
            for subject in subject_words:
                if subject in word:
                    subjects.append(subject)
        setting_words = ['outdoor', 'indoor', 'street', 'park', 'room', 'kitchen', 'garden', 'beach', 'forest']
        for word in words:
            for setting in setting_words:
                if setting in word:
                    settings.append(setting)
        color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange', 'purple', 'colorful']
        for word in words:
            for color in color_words:
                if color in word:
                    colors.append(color)
        action_words = ['sitting', 'standing', 'walking', 'running', 'playing', 'eating', 'sleeping', 'looking']
        for word in words:
            for action in action_words:
                if action in word:
                    actions.append(action)
        description_parts = []
        description_parts.append(f"This image presents {base_caption.lower()}.")
        if subjects:
            unique_subjects = list(set(subjects))
            if len(unique_subjects) == 1:
                description_parts.append(f"The primary subject is a {unique_subjects[0]}, which serves as the focal point of the composition.")
            else:
                description_parts.append(f"The composition features multiple elements including {', '.join(unique_subjects[:-1])} and {unique_subjects[-1]}, creating a diverse visual narrative.")
        if settings:
            setting_context = settings[0]
            if 'outdoor' in setting_context or any(s in setting_context for s in ['street', 'park', 'garden', 'beach', 'forest']):
                description_parts.append(f"The scene takes place in an {setting_context} environment, where natural lighting and spatial elements contribute to the overall atmosphere.")
            else:
                description_parts.append(f"The {setting_context} setting provides an intimate backdrop that frames the subject matter effectively.")
        if colors:
            unique_colors = list(set(colors))
            description_parts.append(f"The color palette features {', '.join(unique_colors)}, which enhances the visual appeal and creates depth in the image.")
        else:
            description_parts.append("The visual composition demonstrates careful attention to contrast, texture, and spatial relationships.")
        if actions:
            action_context = actions[0]
            description_parts.append(f"The captured moment shows the subject {action_context}, creating a sense of life and movement within the frame.")
        description_parts.append("The photographer has skillfully captured this moment with attention to detail, perspective, and timing, resulting in an image that communicates both visual information and emotional resonance.")
        return ' '.join(description_parts)
    elif style == "detailed":
        words = base_caption.lower().split()
        characters = []
        setting = "mysterious realm"
        time_of_day = "golden hour"
        emotions = []
        objects = []
        colors = []
        actions = []
        if any(word in ' '.join(words) for word in ['person', 'man', 'woman', 'child', 'people']):
            characters.append("a solitary figure whose presence commands attention")
        if any(word in ' '.join(words) for word in ['dog', 'cat', 'animal']):
            characters.append("a faithful companion whose eyes hold ancient wisdom")
        if any(word in ' '.join(words) for word in ['group', 'crowd', 'family']):
            characters.append("a gathering of souls, each with their own untold stories")
        if any(word in ' '.join(words) for word in ['park', 'garden', 'outdoor']):
            setting = "a verdant sanctuary where nature weaves its eternal tapestry"
        elif any(word in ' '.join(words) for word in ['street', 'city', 'urban']):
            setting = "the bustling arteries of urban life where stories intersect"
        elif any(word in ' '.join(words) for word in ['beach', 'water', 'ocean']):
            setting = "the timeless boundary between earth and sea"
        elif any(word in ' '.join(words) for word in ['room', 'house', 'indoor']):
            setting = "an intimate chamber where private moments unfold"
        elif any(word in ' '.join(words) for word in ['forest', 'tree', 'nature']):
            setting = "an ancient woodland cathedral where light filters through emerald canopies"
        if any(word in ' '.join(words) for word in ['bright', 'sunny', 'light']):
            time_of_day = "radiant daylight that illuminates every detail with crystal clarity"
        elif any(word in ' '.join(words) for word in ['dark', 'night', 'evening']):
            time_of_day = "the mysterious embrace of twilight"
        elif any(word in ' '.join(words) for word in ['morning', 'dawn']):
            time_of_day = "the hopeful promise of a new dawn"
        if any(word in ' '.join(words) for word in ['happy', 'smiling', 'joy']):
            emotions.append("unbridled joy that radiates from every corner of the frame")
        elif any(word in ' '.join(words) for word in ['calm', 'peaceful', 'serene']):
            emotions.append("profound tranquility that speaks to the soul's deepest longings")
        elif any(word in ' '.join(words) for word in ['sad', 'melancholy']):
            emotions.append("bittersweet melancholy that touches the heart")
        else:
            emotions.append("contemplative serenity that invites deep reflection")
        story_paragraphs = []
        story_paragraphs.append(f"In the realm of captured time, where moments become eternal and fleeting seconds transform into lasting memories, we discover {base_caption.lower()}. This image emerges from the vast tapestry of human experience like a jewel catching the light, demanding our attention and rewarding our contemplation with layers of meaning that unfold like the petals of a rare flower. Here, in this frozen instant, we witness not merely a photograph, but a portal into a world where the ordinary transcends its boundaries and becomes extraordinary. The scene before us pulses with life, energy, and story, waiting patiently for the right observer to unlock its secrets and understand its profound message about the nature of existence itself.")
        story_paragraphs.append(f"The narrative unfolds within {setting}, where every element has been arranged by the invisible hand of destiny to create this perfect moment. The environment itself becomes a character in our story, its presence as vital and compelling as any human figure that might grace the frame. During {time_of_day}, the world takes on a magical quality that transforms the mundane into the magnificent. Light becomes a painter's brush, shadow becomes a poet's metaphor, and space itself becomes a stage upon which this drama plays out. The setting whispers stories of all who have passed through this space before, creating a palimpsest of experience where past, present, and future converge in a single, powerful instant. Here, geography becomes biography, and landscape becomes the silent narrator of countless untold tales.")
        if characters:
            story_paragraphs.append(f"At the heart of our visual narrative stands {characters[0] if characters else 'the central element'}, whose presence anchors the entire composition and provides the emotional core around which all other elements orbit. This figure embodies the universal human experience, serving as both individual and archetype, specific and universal, personal and collective. Their posture, expression, and positioning within the frame speak volumes about their relationship with the world around them, revealing layers of personality, history, and aspiration that extend far beyond what the camera can capture. In their eyes, we glimpse the weight of experience and the lightness of hope, the burden of memory and the gift of presence. They become our guide through this visual journey, our companion in discovery, and our mirror in recognition. Through them, we see not just a single person, but a reflection of all humanity's struggles, triumphs, dreams, and quiet moments of grace.")
        else:
            story_paragraphs.append(f"Though human figures may not dominate this frame, the image pulses with an inherent humanity that speaks to our deepest connections with the world around us. Every element within the composition becomes imbued with character and personality, transforming inanimate objects into vessels of meaning and memory. The absence of obvious human presence paradoxically makes the human element more profound, as we project ourselves into the scene and imagine our own stories unfolding within this carefully crafted moment. This becomes a meditation on place and space, on the way environments hold and shape human experience even when we cannot see the humans themselves. The scene invites us to become the missing character, to step into the frame and complete the story with our own presence and imagination.")
        story_paragraphs.append(f"Permeating every pixel of this captured moment is {emotions[0] if emotions else 'a complex emotional landscape'}, creating an atmosphere that transcends the visual and enters the realm of the felt experience. This emotional quality transforms the image from mere documentation into art, from simple recording into poetry. The mood suffuses the entire composition like morning mist, touching everything it encounters and transforming the ordinary into the sublime. We feel this emotion not just through observation but through participation, as if the photograph has the power to transport us directly into the moment and make us participants rather than mere viewers. The emotional resonance creates ripples that extend far beyond the frame, connecting this single instant to the vast ocean of human feeling and experience that unites us all across time and space.")
        story_paragraphs.append(f"Like a master composer orchestrating a symphony of visual elements, the photographer has created a composition where every component plays its part in perfect harmony. Lines lead the eye on carefully choreographed journeys through the frame, colors create emotional resonances that speak directly to the subconscious, and textures invite tactile imagination that makes us want to reach out and touch the surface of the captured world. The interplay of light and shadow becomes a dialogue between hope and mystery, revelation and concealment, clarity and dream. Depth of field creates layers of meaning, with some elements sharp and immediate while others fade into soft suggestions of possibility. The framing itself becomes a statement about perspective and choice, about what to include and what to leave to imagination. Every technical decision serves the greater purpose of storytelling, creating a visual language that speaks directly to the heart without need for translation.")
        story_paragraphs.append(f"This photograph transcends its immediate subject matter to touch upon the great universal themes that have fascinated humanity since the dawn of consciousness: the passage of time, the nature of beauty, the complexity of relationships, the search for meaning, and the eternal dance between permanence and change. Within this single frame, we find echoes of every story ever told, every poem ever written, every song ever sung. The image becomes a mirror in which we see not just the captured moment, but our own hopes, fears, dreams, and memories reflected back at us. It reminds us that every moment contains within it the seeds of eternity, that every ordinary instant has the potential to become extraordinary if viewed with the right eyes and an open heart. This is the magic of photography at its finest: the ability to reveal the profound within the mundane.")
        story_paragraphs.append(f"Behind this remarkable image stands an artist whose vision extends far beyond technical proficiency to encompass a deep understanding of human nature and visual storytelling. The photographer has not merely pointed a camera and pressed a button, but has engaged in an act of profound seeing that transforms observation into revelation. Their eye has found the perfect moment when all elements align to create something greater than the sum of their parts. The artist's choices – of timing, angle, focal length, and processing – reveal a sensibility attuned to the poetry of everyday life and the extraordinary moments that hide within ordinary experience. This image stands as testament to the photographer's ability to see what others might miss and to preserve it for posterity.")
        story_paragraphs.append(f"As we step back from this intensive contemplation of a single captured moment, we carry with us not just the memory of what we have seen, but a renewed appreciation for the power of visual storytelling and the profound beauty that surrounds us in every moment of our lives. This photograph will continue to speak to viewers long after we have turned away, its message evolving and deepening with each new pair of eyes that encounters it. It stands as both a document of a specific moment in time and a timeless meditation on the human condition, reminding us that art's greatest gift is its ability to make us see the world with fresh eyes and feel our connection to the larger tapestry of existence.")
        return '\n\n'.join(story_paragraphs)
    return base_caption

def generate_caption(image, style="balanced"):
    global model, processor, tokenizer
    try:
        inputs = processor(image, return_tensors="pt")
        base_gen_kwargs = {"max_length": 50, "num_beams": 5, "do_sample": True, "temperature": 0.7, "repetition_penalty": 1.1}
        with torch.no_grad():
            generated_ids = model.generate(inputs.pixel_values, **base_gen_kwargs)
        base_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if not base_caption.endswith('.'):
            base_caption += '.'
        enhanced_caption = enhance_caption_with_style(base_caption, style)
        return enhanced_caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

@app.route('/')
def index():
    return "<h2>Unified AI Image Caption Generator (Patent Version)</h2><p>Use /api/generate_caption to POST images.</p>"

@app.route('/api/generate_caption', methods=['POST'])
def api_generate_caption():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        style = request.form.get('style', 'balanced')
        image = Image.open(file.stream).convert('RGB')
        caption = generate_caption(image, style)
        word_count = len(caption.split())
        return jsonify({'caption': caption, 'style': style, 'word_count': word_count, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_status')
def api_model_status():
    return jsonify({'loaded': model_loaded, 'pytorch_version': torch.__version__ if torch else None, 'styles_available': ['short', 'balanced', 'detailed']})

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model_loaded, 'pytorch_available': torch is not None, 'enhanced_styles': True})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    print("Loading AI model...")
    load_ai_model()
    print("Starting unified web server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
