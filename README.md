# 🖼️ Caption to Image — Intelligent Image Captioning System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![Flask](https://img.shields.io/badge/framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced AI-powered computer vision application that generates rich, descriptive natural language captions for any image in three distinct styles: **Short Labels**, **Balanced Descriptions**, and **Full Narrative Stories**. Built using Google's **Vision Transformer (ViT)** and OpenAI's **GPT-2** with an interactive **Flask** web dashboard.

---

## 🌟 Demo & Caption Styles

Upload any image ➜ choose a generation style ➜ get an AI caption in seconds.

| Style | Output Description |
|---|---|
| **Short** | `"A person sitting on a wooden park bench feeding pigeons."` *(8–10 words, ideal for Alt-Text)* |
| **Balanced** | Structured multi-sentence paragraph covering subjects, setting, lighting, actions, and colors. |
| **Detailed Story** | Immersive multi-paragraph narrative covering background lore, atmosphere, emotional depth, and artistic composition. |

---

## 🧠 Architecture Workflow

```
Image Upload
    ↓
Vision Transformer (ViT - google/vit-base-patch16-224-in21k)
  → Deconstructs image into 16×16 visual patches
  → Encodes visual features into 768-dimensional token embeddings
    ↓
VisionEncoderDecoder (HuggingFace)
  → Cross-attention bridging visual tokens and GPT-2 language decoder
    ↓
GPT-2 Decoder (Beam Search, n=5)
  → Generates high-confidence foundational caption
    ↓
Style Transformation Engine
  → Short:    Core subject & action extraction
  → Balanced: Environment & contextual expansion
  → Detailed: Atmospheric storytelling & composition breakdown
    ↓
Interactive Web Dashboard (Flask + HTML5 / CSS3 / JavaScript)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Image Encoder** | Google Vision Transformer (`google/vit-base-patch16-224-in21k`) |
| **Caption Model** | Vision-Encoder-Decoder (`nlpconnect/vit-gpt2-image-captioning`) |
| **Language Decoder** | OpenAI GPT-2 (`gpt2`) |
| **Deep Learning** | PyTorch + HuggingFace Transformers |
| **Web Server** | Flask + Flask-CORS |
| **Frontend UI** | Responsive Modern Glassmorphic Dashboard (HTML5, Vanilla CSS, JS) |

---

## 📂 Project Structure

```
Caption-to-Image/
├── app.py                       # Main Flask web server & REST API
├── inference.py                 # Custom model inference & style generation wrapper
├── config.py                    # Hyperparameters & model configuration
├── story_model.py               # Long-form narrative generation logic
├── setup.py                     # Package setup
├── requirements.txt             # Python dependencies
├── start_app.bat                # 1-click Windows launcher
├── README.md                    # Complete documentation
│
├── templates/
│   ├── index.html               # Main interactive dashboard
│   └── enhanced_index.html      # Glassmorphic UI template
│
├── static/
│   ├── style.css                # Dark mode design system & animations
│   └── main.js                  # Image preview, drag-and-drop & AJAX runner
│
└── uploads/                     # Temporary upload directory (.gitkeep)
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/harshrameshnerkar/Caption-to-Image.git
cd Caption-to-Image
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: On first run, HuggingFace `transformers` will automatically download the pre-trained ViT+GPT2 model weights (~900 MB). Subsequent runs load instantly from local cache.

### 3. Launch the Application
```bash
python app.py
```
*(Or double-click `start_app.bat` on Windows)*

Open your browser at **[http://localhost:5000](http://localhost:5000)**.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web User Interface |
| `POST` | `/api/generate_caption` | Generates a caption from an uploaded image (`image`, `style`) |
| `GET` | `/api/model_status` | Returns model loading status |
| `GET` | `/health` | Service health check |

---

## ⚡ Performance

| Hardware | Average Inference Time |
|---|---|
| **CPU Only** | ~8–15 seconds |
| **NVIDIA GPU (CUDA)** | ~1–2 seconds |

The model automatically detects and uses CUDA acceleration if available.

---

## 👤 Author

**Harsh Nerkar**
- GitHub: [@harshrameshnerkar](https://github.com/harshrameshnerkar)
- Email: [harshrameshnerkar@gmail.com](mailto:harshrameshnerkar@gmail.com)

---

## 📄 License
This project is licensed under the MIT License.
