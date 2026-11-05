"""
CaptionAI Studio — Bidirectional Multimodal AI Platform.
Features:
1. Caption to Image (Text-to-Image Generation with styles and aspect ratios)
2. Image to Caption (Vision-to-Text with Short, Balanced, Detailed Story & Social styles + TTS)
3. Multimodal Mix & Remix (Transform images via creative caption steering)
4. Dataset Management & Model Training Hub
"""
import os
import time
import base64
import traceback
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename
from gtts import gTTS

# Import core modules
from caption_to_image import caption_to_image_engine, STYLE_PRESETS
from remix_engine import remix_engine
from dataset_manager import dataset_manager
from training_engine import training_engine

# Initialize Flask app
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GENERATED_FOLDER = os.path.join(UPLOAD_FOLDER, "generated")
AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, "audio")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")

for folder in [UPLOAD_FOLDER, GENERATED_FOLDER, AUDIO_FOLDER, DATASET_DIR]:
    os.makedirs(folder, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Optional ViT-GPT2 Model Initialization
model = None
processor = None
tokenizer = None
model_loaded = False
model_loading_error = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def try_load_vit_gpt2():
    """Attempts to load local or cached VisionEncoderDecoder model if available."""
    global model, processor, tokenizer, model_loaded, model_loading_error
    try:
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
        import torch

        print("Checking Vision Transformer + GPT-2 model...")
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        tokenizer.pad_token = tokenizer.eos_token
        model.config.decoder_start_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        model_loaded = True
        print("✅ ViT-GPT2 Model loaded successfully!")
    except Exception as e:
        model_loading_error = str(e)
        model_loaded = False
        print(f"ℹ️ ViT-GPT2 not pre-cached ({e}). Intelligent vision fallback engine active.")


def analyze_image_fallback(image: Image.Image) -> str:
    """Heuristic visual analysis when heavy neural weights are offline or downloading."""
    width, height = image.size
    aspect = width / height if height > 0 else 1.0

    # Sample color distribution
    img_small = image.resize((50, 50)).convert("RGB")
    colors = img_small.getcolors(2500)
    dominant_color = max(colors, key=lambda item: item[0])[1] if colors else (128, 128, 128)

    r, g, b = dominant_color
    if r > 180 and g > 180 and b > 180:
        tone = "bright, high-key studio scene"
    elif r < 60 and g < 60 and b < 60:
        tone = "dark, moody atmospheric composition"
    elif b > r and b > g:
        tone = "cool-toned blue setting featuring open sky or water"
    elif g > r and g > b:
        tone = "lush green natural environment with flora"
    elif r > g and r > b:
        tone = "warm, golden-hued artistic scene"
    else:
        tone = "balanced harmonic composition"

    format_desc = "panoramic landscape view" if aspect > 1.3 else ("vertical portrait composition" if aspect < 0.8 else "balanced square-proportioned view")
    return f"A compelling photograph showcasing a {tone} in a {format_desc} with distinct focal depth"


def format_caption(base_text: str, style: str = "balanced") -> dict:
    """Formats the base caption into rich output styles."""
    words = base_text.replace(".", "").split()

    if style == "short":
        # Concise 8-12 word caption
        cleaned = " ".join(words[:11]).capitalize()
        if not cleaned.endswith("."):
            cleaned += "."
        caption = cleaned
    elif style == "detailed":
        caption = (
            f"An evocative and intricately detailed photograph capturing {base_text.lower().rstrip('.')}. "
            "The visual hierarchy guides the viewer through layers of foreground interest and subtle background textures, "
            "bathed in cinematic ambient illumination that accentuates tactile materials and natural dimensionality. "
            "The artistic framing and disciplined depth of field evoke an atmospheric sense of narrative storytelling."
        )
    elif style == "social":
        tags = ["#AIArt", "#Photography", "#VisualStorytelling", "#InstaGood", "#Creativity", "#DigitalVision"]
        caption = f"✨ Captured beauty: {base_text.rstrip('.')}! Every detail comes alive through the lens. 📸\n\n{' '.join(tags)}"
    else:  # balanced
        caption = (
            f"A vibrant and clear view showing {base_text.lower().rstrip('.')}. "
            "The scene exhibits natural lighting with defined contours and a harmonious color palette that lends realism and depth."
        )

    return {
        "caption": caption,
        "style": style,
        "alt_text": f"Image showing {base_text.lower().rstrip('.')}",
        "tags": [w.strip(",.").lower() for w in words if len(w) > 4][:6]
    }


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main Web Studio"""
    return render_template('enhanced_index.html', styles=STYLE_PRESETS)


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "CaptionAI Multimodal Studio",
        "model_loaded": model_loaded,
        "active_features": ["caption_to_image", "image_to_caption", "remix", "dataset_hub"]
    })


# --- 1. Caption to Image Endpoint ---
@app.route('/api/generate_image', methods=['POST'])
def api_generate_image():
    try:
        data = request.get_json() or {}
        prompt = data.get('prompt', '').strip()
        style = data.get('style', 'photorealistic')
        aspect_ratio = data.get('aspect_ratio', '1:1')
        negative_prompt = data.get('negative_prompt', '')
        seed = data.get('seed')

        if not prompt:
            return jsonify({"success": False, "error": "Please provide a prompt/caption."}), 400

        result = caption_to_image_engine.generate(
            prompt=prompt,
            style=style,
            aspect_ratio=aspect_ratio,
            seed=seed,
            negative_prompt=negative_prompt
        )

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# --- 2. Image to Caption Endpoint ---
@app.route('/api/generate_caption', methods=['POST'])
def api_generate_caption():
    try:
        style = request.form.get('style', 'balanced')
        image = None

        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(f"upload_{int(time.time())}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image = Image.open(filepath).convert("RGB")
        elif request.is_json:
            json_data = request.get_json()
            image_b64 = json_data.get('image_base64', '')
            style = json_data.get('style', style)
            if image_b64:
                header, encoded = image_b64.split(",", 1) if "," in image_b64 else ("", image_b64)
                image_data = base64.b64decode(encoded)
                image = Image.open(BytesIO(image_data)).convert("RGB")

        if image is None:
            return jsonify({"success": False, "error": "No valid image provided."}), 400

        # Generate base caption
        base_caption = ""
        if model_loaded and model and processor and tokenizer:
            import torch
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                generated_ids = model.generate(inputs.pixel_values, max_length=50, num_beams=5)
            base_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        if not base_caption:
            base_caption = analyze_image_fallback(image)

        formatted = format_caption(base_caption, style=style)
        formatted["success"] = True
        return jsonify(formatted)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# --- 3. Multimodal Remix Endpoint ---
@app.route('/api/remix', methods=['POST'])
def api_remix():
    try:
        data = request.get_json() or {}
        base_caption = data.get('base_caption', '').strip()
        user_modification = data.get('user_modification', '').strip()
        target_style = data.get('target_style', 'photorealistic')
        aspect_ratio = data.get('aspect_ratio', '1:1')
        seed = data.get('seed')

        if not base_caption:
            return jsonify({"success": False, "error": "Base caption is required for remix."}), 400

        result = remix_engine.remix_image(
            base_caption=base_caption,
            user_modification=user_modification,
            target_style=target_style,
            aspect_ratio=aspect_ratio,
            seed=seed
        )

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# --- 4. Text-to-Speech (TTS) Endpoint ---
@app.route('/api/tts', methods=['POST'])
def api_tts():
    try:
        data = request.get_json() or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"success": False, "error": "Text is required for TTS."}), 400

        tts = gTTS(text=text, lang='en', slow=False)
        filename = f"tts_{int(time.time())}.mp3"
        filepath = os.path.join(AUDIO_FOLDER, filename)
        tts.save(filepath)

        return jsonify({
            "success": True,
            "audio_url": f"/uploads/audio/{filename}"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --- 5. Dataset Endpoints ---
@app.route('/api/dataset/samples', methods=['GET'])
def api_dataset_samples():
    samples = dataset_manager.get_dataset_samples()
    stats = dataset_manager.get_stats()
    return jsonify({"success": True, "samples": samples, "stats": stats})


@app.route('/api/dataset/download', methods=['POST'])
def api_dataset_download():
    result = dataset_manager.download_seed_dataset()
    return jsonify(result)


# --- 6. Training Endpoints ---
@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    data = request.get_json() or {}
    model_type = data.get('model_type', 'captioning')
    epochs = int(data.get('epochs', 5))
    lr = float(data.get('learning_rate', 5e-5))

    res = training_engine.start_training(model_type=model_type, epochs=epochs, learning_rate=lr)
    return jsonify(res)


@app.route('/api/train/status', methods=['GET'])
def api_train_status():
    status = training_engine.get_status()
    return jsonify(status)


@app.route('/api/train/stop', methods=['POST'])
def api_train_stop():
    res = training_engine.stop_training()
    return jsonify(res)


# --- Static and Asset Serving ---
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/dataset/images/<path:filename>')
def serve_dataset_images(filename):
    return send_from_directory(DATASET_DIR, filename)


if __name__ == '__main__':
    # Initialize seed dataset on startup if not present
    dataset_manager.download_seed_dataset()
    print("=" * 60)
    print(" 🚀 CaptionAI Multimodal Studio Running at http://127.0.0.1:5000")
    print(" 1. Caption to Image (Text-to-Image AI Diffusion)")
    print(" 2. Image to Caption (Vision-to-Text Multi-Style + TTS)")
    print(" 3. Multimodal Mix & Remix Studio")
    print(" 4. Dataset Manager & Model Training Hub")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)
