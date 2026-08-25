# CaptionAI — Intelligent Image Captioning System

An AI-powered web application that generates natural language captions for any image in three distinct styles: short labels, balanced descriptions, and rich story narratives. Built on Google's Vision Transformer (ViT) and OpenAI's GPT-2 with a Flask backend.

---

## Demo

Upload any image → choose a style → get an AI-generated caption in seconds.

| Style | Output |
|---|---|
| **Short** | `"person sitting on park bench feeding pigeons"` (8–10 words) |
| **Balanced** | Multi-sentence paragraph covering subjects, setting, colors, actions |
| **Detailed** | Full narrative story — characters, setting, emotion, artistic analysis |

---

## How It Works

```
Image Upload
    ↓
ViT (Vision Transformer)
  → Splits image into 16×16 patches
  → Encodes patches as 768-dim visual tokens
    ↓
VisionEncoderDecoderModel (HuggingFace)
  → Cross-attention between visual tokens and language decoder
    ↓
GPT-2 Decoder (Beam Search, n=5)
  → Generates base caption
    ↓
Style Enhancement Layer
  → Short:    keyword extraction → 8–10 word description
  → Balanced: template expansion → multi-line paragraph
  → Detailed: narrative generation → story format
    ↓
Web UI (Flask + HTML/CSS/JavaScript)
```

---

## Tech Stack

| Component | Library / Model |
|---|---|
| Image encoder | `google/vit-base-patch16-224-in21k` |
| Caption model | `nlpconnect/vit-gpt2-image-captioning` |
| Language decoder | GPT-2 (`openai-community/gpt2`) |
| Web framework | Flask + Flask-CORS |
| Frontend | Vanilla HTML / CSS / JS (Responsive UI) |
| ML framework | PyTorch + HuggingFace Transformers |

---

## Project Structure

```
5_CAPTIONTOIMAGE/
├── app.py                       # Main Flask web app entry point
├── enhanced_flask_app.py        # Flask app
├── inference.py                 # Custom model inference wrapper
├── templates/
│   ├── enhanced_index.html      # Professional web UI
│   └── index.html
├── static/
│   ├── style.css
│   └── main.js
├── uploads/                     # Temp uploaded images
├── requirements.txt
├── start_app.bat                # Windows launch script
└── README.md
```

---

## Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `transformers` will download the ViT+GPT2 model (~900 MB) on first run. Subsequent runs use the local cache.

### 2. Run the Web App

```bash
python app.py
```

*Alternatively:*
```bash
python enhanced_flask_app.py
```
*Or double-click `start_app.bat`.*

### 3. Open in Browser

```
http://localhost:5000
```

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/generate_caption` | Generate caption for uploaded image |
| `GET` | `/api/model_status` | Check if model is loaded |
| `GET` | `/health` | Health check |

### POST `/api/generate_caption`

**Request** (multipart/form-data):
```
image  : image file (PNG/JPG/JPEG/GIF/BMP/WEBP, max 16MB)
style  : "short" | "balanced" | "detailed"
```

**Response**:
```json
{
  "caption": "...",
  "style": "balanced",
  "word_count": 87,
  "success": true
}
```

---

## Caption Styles

### Short
Extracts key subjects, actions, and objects from the base model output. Returns an 8–10 word description suitable for alt-text or quick labeling.

### Balanced
Builds a structured paragraph by analyzing:
- Primary subjects and their roles
- Setting and environment type
- Color palette references
- Actions and behavioral cues
- Artistic/compositional assessment

### Detailed
Generates an immersive narrative structured into rich thematic sections:
1. Discovery / Opening
2. Setting the Scene
3. Character Development
4. Emotional Journey
5. Visual Symphony (composition analysis)
6. Universal Themes
7. Legacy / Conclusion

---

## Performance

| Hardware | Avg Generation Time |
|---|---|
| CPU only | 10–25 seconds |
| NVIDIA GPU (CUDA) | 1–3 seconds |

The model automatically uses CUDA if available:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Requirements

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended)
- Internet connection on first run (model download)

---

## Author

**Harsh Ramesh Nerkar**  
B.Tech CSE 2026 | AI/ML Engineer  
Email: harshrameshnerkar@gmail.com
