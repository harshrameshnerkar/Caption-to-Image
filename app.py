"""
AI Image Caption Generator - Flask Web Application
Provides: Short (8-10 words), Balanced (detailed description paragraph), Detailed (story format)
"""
import os
import base64
import io
import shutil
import traceback
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Global variables for model components
model = None
processor = None
tokenizer = None
model_loaded = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_ai_model():
    """Load the AI model components"""
    global model, processor, tokenizer, model_loaded
    
    try:
        print("Loading AI model...")
        
        # Load model
        model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        
        # Load processor and tokenizer
        processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # Set special tokens
        tokenizer.pad_token = tokenizer.eos_token
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        
        model_loaded = True
        print("✅ AI model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False


def enhance_caption_with_style(base_caption, style="balanced"):
    """Enhance the base caption based on the selected style"""
    
    if style == "short":
        # Create precise 8-10 word description
        words = base_caption.lower().split()
        
        # Extract key elements for precise description
        subjects = []
        actions = []
        locations = []
        objects = []
        
        # Enhanced subject detection
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
        
        # Build precise 8-10 word description
        description_parts = []
        
        # Add primary subject
        if subjects:
            description_parts.append(subjects[0])
        
        # Add action if present
        if actions:
            description_parts.append(actions[0])
        
        # Add key objects/location
        if objects:
            description_parts.extend(objects[:2])
        elif locations:
            description_parts.append(locations[0])
        
        # Fill to 8-10 words with additional context
        remaining_words = [w for w in words if len(w) > 3 and w not in ['with', 'and', 'the', 'in', 'on', 'at']]
        for word in remaining_words:
            if len(description_parts) < 8 and word not in ' '.join(description_parts):
                description_parts.append(word)
        
        # Ensure we have 8-10 words
        if len(description_parts) < 8:
            additional_words = [w for w in words if len(w) > 2 and w not in description_parts]
            description_parts.extend(additional_words[:8-len(description_parts)])
        
        final_description = ' '.join(description_parts[:10])
        
        if len(description_parts) >= 8:
            return final_description
        else:
            return f"{base_caption.split('.')[0][:60]}..."
    
    elif style == "balanced":
        # Create precise paragraph description analyzing image features and elements
        words = base_caption.lower().split()
        
        # Analyze the base caption for key elements
        subjects = []
        settings = []
        colors = []
        actions = []
        
        # Subject detection
        subject_words = ['person', 'man', 'woman', 'child', 'dog', 'cat', 'animal', 'car', 'building', 'tree', 'flower']
        for word in words:
            for subject in subject_words:
                if subject in word:
                    subjects.append(subject)
        
        # Setting detection
        setting_words = ['outdoor', 'indoor', 'street', 'park', 'room', 'kitchen', 'garden', 'beach', 'forest']
        for word in words:
            for setting in setting_words:
                if setting in word:
                    settings.append(setting)
        
        # Color detection
        color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange', 'purple', 'colorful']
        for word in words:
            for color in color_words:
                if color in word:
                    colors.append(color)
        
        # Action detection
        action_words = ['sitting', 'standing', 'walking', 'running', 'playing', 'eating', 'sleeping', 'looking']
        for word in words:
            for action in action_words:
                if action in word:
                    actions.append(action)
        
        # Build comprehensive paragraph
        description_parts = []
        
        # Start with the base observation
        description_parts.append(f"This image presents {base_caption.lower()}.")
        
        # Add subject analysis
        if subjects:
            unique_subjects = list(set(subjects))
            if len(unique_subjects) == 1:
                description_parts.append(f"The primary subject is a {unique_subjects[0]}, which serves as the focal point of the composition.")
            else:
                description_parts.append(f"The composition features multiple elements including {', '.join(unique_subjects[:-1])} and {unique_subjects[-1]}, creating a diverse visual narrative.")
        
        # Add setting context
        if settings:
            setting_context = settings[0]
            if 'outdoor' in setting_context or any(s in setting_context for s in ['street', 'park', 'garden', 'beach', 'forest']):
                description_parts.append(f"The scene takes place in an {setting_context} environment, where natural lighting and spatial elements contribute to the overall atmosphere.")
            else:
                description_parts.append(f"The {setting_context} setting provides an intimate backdrop that frames the subject matter effectively.")
        
        # Add visual elements analysis
        if colors:
            unique_colors = list(set(colors))
            description_parts.append(f"The color palette features {', '.join(unique_colors)}, which enhances the visual appeal and creates depth in the image.")
        else:
            description_parts.append("The visual composition demonstrates careful attention to contrast, texture, and spatial relationships.")
        
        # Add behavioral/action context
        if actions:
            action_context = actions[0]
            description_parts.append(f"The captured moment shows the subject {action_context}, creating a sense of life and movement within the frame.")
        
        # Add technical/artistic appreciation
        description_parts.append("The photographer has skillfully captured this moment with attention to detail, perspective, and timing, resulting in an image that communicates both visual information and emotional resonance.")
        
        return ' '.join(description_parts)
    
    elif style == "detailed":
        # Create comprehensive narrative story about the image
        words = base_caption.lower().split()
        
        characters = []
        setting = "mysterious realm"
        time_of_day = "golden hour"
        emotions = []
        
        # Character analysis
        if any(word in ' '.join(words) for word in ['person', 'man', 'woman', 'child', 'people']):
            characters.append("a solitary figure whose presence commands attention")
        if any(word in ' '.join(words) for word in ['dog', 'cat', 'animal']):
            characters.append("a faithful companion whose eyes hold ancient wisdom")
        if any(word in ' '.join(words) for word in ['group', 'crowd', 'family']):
            characters.append("a gathering of souls, each with their own untold stories")
        
        # Setting analysis
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
        
        # Time and atmosphere analysis
        if any(word in ' '.join(words) for word in ['bright', 'sunny', 'light']):
            time_of_day = "radiant daylight that illuminates every detail with crystal clarity"
        elif any(word in ' '.join(words) for word in ['dark', 'night', 'evening']):
            time_of_day = "the mysterious embrace of twilight"
        elif any(word in ' '.join(words) for word in ['morning', 'dawn']):
            time_of_day = "the hopeful promise of a new dawn"
        
        # Emotional analysis
        if any(word in ' '.join(words) for word in ['happy', 'smiling', 'joy']):
            emotions.append("unbridled joy that radiates from every corner of the frame")
        elif any(word in ' '.join(words) for word in ['calm', 'peaceful', 'serene']):
            emotions.append("profound tranquility that speaks to the soul's deepest longings")
        elif any(word in ' '.join(words) for word in ['sad', 'melancholy']):
            emotions.append("bittersweet melancholy that touches the heart")
        else:
            emotions.append("contemplative serenity that invites deep reflection")
        
        story_paragraphs = []
        
        # Opening - The Discovery
        story_paragraphs.append(f"In the realm of captured time, where moments become eternal and fleeting seconds transform into lasting memories, we discover {base_caption.lower()}. This image emerges from the vast tapestry of human experience like a jewel catching the light, demanding our attention and rewarding our contemplation with layers of meaning that unfold like the petals of a rare flower. Here, in this frozen instant, we witness not merely a photograph, but a portal into a world where the ordinary transcends its boundaries and becomes extraordinary. The scene before us pulses with life, energy, and story, waiting patiently for the right observer to unlock its secrets and understand its profound message about the nature of existence itself.")
        
        # Setting the Stage
        story_paragraphs.append(f"The narrative unfolds within {setting}, where every element has been arranged by the invisible hand of destiny to create this perfect moment. The environment itself becomes a character in our story, its presence as vital and compelling as any human figure that might grace the frame. During {time_of_day}, the world takes on a magical quality that transforms the mundane into the magnificent. Light becomes a painter's brush, shadow becomes a poet's metaphor, and space itself becomes a stage upon which this drama plays out. The setting whispers stories of all who have passed through this space before, creating a palimpsest of experience where past, present, and future converge in a single, powerful instant.")
        
        # Character Development
        if characters:
            story_paragraphs.append(f"At the heart of our visual narrative stands {characters[0]}, whose presence anchors the entire composition and provides the emotional core around which all other elements orbit. This figure embodies the universal human experience, serving as both individual and archetype, specific and universal, personal and collective. Their posture, expression, and positioning within the frame speak volumes about their relationship with the world around them, revealing layers of personality, history, and aspiration that extend far beyond what the camera can capture.")
        else:
            story_paragraphs.append(f"Though human figures may not dominate this frame, the image pulses with an inherent humanity that speaks to our deepest connections with the world around us. Every element within the composition becomes imbued with character and personality, transforming inanimate objects into vessels of meaning and memory. The absence of obvious human presence paradoxically makes the human element more profound, as we project ourselves into the scene and imagine our own stories unfolding within this carefully crafted moment.")
        
        # The Emotional Journey
        story_paragraphs.append(f"Permeating every pixel of this captured moment is {emotions[0]}, creating an atmosphere that transcends the visual and enters the realm of the felt experience. This emotional quality transforms the image from mere documentation into art, from simple recording into poetry. The mood suffuses the entire composition like morning mist, touching everything it encounters and transforming the ordinary into the sublime.")
        
        # Visual Symphony
        story_paragraphs.append(f"Like a master composer orchestrating a symphony of visual elements, the composition demonstrates careful attention to detail, line, and contrast where every component plays its part in perfect harmony. Lines lead the eye on carefully choreographed journeys through the frame, colors create emotional resonances that speak directly to the subconscious, and textures invite tactile imagination. The interplay of light and shadow becomes a dialogue between hope and mystery, revelation and concealment.")
        
        # Universal Themes
        story_paragraphs.append(f"This photograph transcends its immediate subject matter to touch upon the great universal themes that have fascinated humanity: the passage of time, the nature of beauty, the search for meaning, and the eternal dance between permanence and change. Within this single frame, we find echoes of stories that remind us every ordinary instant has the potential to become extraordinary if viewed with the right eyes and an open heart.")
        
        # Conclusion
        story_paragraphs.append(f"As we step back from this contemplation of a single captured moment, we carry with us not just the memory of what we have seen, but a renewed appreciation for the power of visual storytelling and the profound beauty that surrounds us in every moment of our lives. It stands as both a document of a specific moment in time and a timeless meditation on the human condition.")
        
        return '\n\n'.join(story_paragraphs)
    
    return base_caption


def generate_caption(image, style="balanced"):
    """Generate caption for an image with enhanced styling"""
    global model, processor, tokenizer
    
    try:
        inputs = processor(image, return_tensors="pt")
        
        base_gen_kwargs = {
            "max_length": 50,
            "num_beams": 5,
            "do_sample": True,
            "temperature": 0.7,
            "repetition_penalty": 1.1
        }
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.pixel_values,
                **base_gen_kwargs
            )
        
        base_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        base_caption = base_caption.strip()
        if not base_caption.endswith('.'):
            base_caption += '.'
        
        enhanced_caption = enhance_caption_with_style(base_caption, style)
        return enhanced_caption
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        return f"Error generating caption: {str(e)}"


# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('enhanced_index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


@app.route('/api/generate_caption', methods=['POST'])
def api_generate_caption():
    """API endpoint to generate captions"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
        
        style = request.form.get('style', 'balanced')
        image = Image.open(file.stream).convert('RGB')
        caption = generate_caption(image, style)
        
        word_count = len(caption.split()) if style == "short" else len(caption.split('\n')) if style == "balanced" else len(caption.split())
        
        return jsonify({
            'caption': caption,
            'style': style,
            'word_count': word_count,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in API: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_status')
def api_model_status():
    """Check if model is loaded"""
    return jsonify({
        'loaded': model_loaded,
        'pytorch_version': torch.__version__ if torch else None,
        'styles_available': ['short', 'balanced', 'detailed']
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'pytorch_available': torch is not None,
        'enhanced_styles': True
    })


if __name__ == '__main__':
    # Create upload directory
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Load AI model
    print("🚀 Starting Enhanced AI Image Caption Generator...")
    load_ai_model()
    
    # Start the Flask app
    print("\n🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔥 Press Ctrl+C to stop the server")
    print("\n✨ Enhanced Features:")
    print("   • Short: Key objects (8-10 words)")
    print("   • Balanced: Detailed description (5-8 lines)")
    print("   • Detailed: Story format narrative")
    
    app.run(host='0.0.0.0', port=5000, debug=True)