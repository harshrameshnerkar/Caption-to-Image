# 🖼️ CaptionAI — Bidirectional Multimodal AI Vision Studio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers)
[![Diffusion](https://img.shields.io/badge/Diffusion-Text--to--Image-purple)](https://huggingface.co/)
[![Flask](https://img.shields.io/badge/framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced bidirectional multimodal artificial intelligence platform that performs **Text-to-Image generation** (Caption to Image), **Image-to-Text captioning** (Image to Caption), **Multimodal Remixing** (Image + Prompt steering), and an interactive **Dataset & Model Training Hub**.

Built with **PyTorch**, **HuggingFace Transformers**, **Vision Transformer (ViT)**, **GPT-2**, **AI Diffusion**, **Text-to-Speech (TTS)**, and an interactive **Glassmorphic Web Dashboard**.

---

## 🌟 Core Multimodal Capabilities

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CaptionAI Multimodal Engine                     │
├──────────────────────────┬─────────────────────────────────────────────┤
│  ✍️ Caption to Image     │  Natural Language Prompt ➜ AI Visual        │
│  🔍 Image to Caption     │  Image ➜ Multi-Style Descriptions + TTS    │
│  🔀 Multimodal Remix     │  Image ➜ Extract Concept ➜ Steer ➜ New Art  │
│  📊 Training & Datasets  │  1-Click Dataset Grabber ➜ Fine-Tuning Hub │
└──────────────────────────┴─────────────────────────────────────────────┘
```

### 1. ✍️ Caption to Image (Text $\rightarrow$ Image)
* Transforms any prompt or caption into a high-resolution visual.
* **Art Style Presets:** *Photorealistic 8K, Cinematic Film, Anime & Manga, Cyberpunk Neon, Epic Fantasy, 3D Pixar / Octane Render, Dreamy Watercolor, and Minimalist Studio*.
* **Aspect Ratios:** `1:1` (Square), `16:9` (Cinema), `9:16` (Mobile/Story), `4:3` (Standard).
* **Zero-GPU Cloud Diffusion with Local Fallback Engine:** Works instantly on any machine without demanding heavy GPU VRAM.

### 2. 🔍 Image to Caption (Image $\rightarrow$ Text)
* Upload any image to extract rich, natural language descriptions in 4 selectable narrative styles:
  * **Short Alt-Text:** 8–12 words, optimized for accessibility and SEO.
  * **Balanced Paragraph:** Covers subjects, actions, environment, lighting, and palette.
  * **Cinematic Story:** Immersive prose highlighting mood, atmosphere, and composition.
  * **Social Media:** Engaging caption with auto-generated hashtags (`#AIArt`, `#Photography`).
* **Text-to-Speech (TTS):** 1-click audio narration to listen to captions.

### 3. 🔀 Multimodal Mix & Remix Studio
* Upload a source image $\rightarrow$ extract its visual concept $\rightarrow$ apply creative user modifications (e.g. *"Set in ancient feudal Japan with cherry blossom rain"*) $\rightarrow$ synthesize an entirely new reimagined AI visual variation.

### 4. 📊 Dataset Manager & Model Training Hub
* **1-Click Dataset Acquisition:** Automated downloader that establishes paired image-caption datasets with images and `metadata.csv`.
* **Live Dataset Inspector:** Visual gallery previewing ground-truth image-caption pairs.
* **Training Telemetry Console:** Background training engine with real-time loss tracking, epoch progress, and checkpoint export.

---

## 🏛 System Architecture

```
                  ┌───────────────────────────────┐
                  │   Interactive Web Studio UI   │
                  │   (Glassmorphic Dark Theme)   │
                  └───────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Caption to Image │    │ Image to Caption │    │  Dataset & Train │
├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│ Style Presets    │    │ ViT-GPT2 Model   │    │ Dataset Manager  │
│ Aspect Ratios    │    │ Multi-Style Gen  │    │ Image Harvester  │
│ Cloud Diffusion  │    │ Fallback Vision  │    │ Training Engine  │
│ Local Canvas Gen │    │ gTTS Audio Engine│    │ Realtime Loss    │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Text-to-Image Diffusion** | High-Speed AI Diffusion Engine + PIL Fallback Canvas |
| **Image-to-Caption Model** | Vision Transformer (`google/vit-base-patch16-224-in21k`) + `GPT-2` |
| **Deep Learning Framework** | PyTorch + HuggingFace Transformers + Accelerate |
| **Text-to-Speech (TTS)** | Google Text-to-Speech (`gTTS`) |
| **Backend API** | Flask + Flask-CORS + Werkzeug |
| **Frontend UI** | Responsive Modern Glassmorphic Studio (HTML5, CSS3, FontAwesome) |

---

## 📂 Project Structure

```
Caption-to-Image/
├── app.py                       # Unified Flask web application & REST API
├── caption_to_image.py          # Text-to-Image generation engine (styles & aspect ratios)
├── remix_engine.py              # Multimodal mix & visual variation engine
├── dataset_manager.py           # Dataset acquisition, download & metadata manager
├── training_engine.py           # Background training orchestrator & loss telemetry
├── test_multimodal.py           # Comprehensive end-to-end test suite
├── requirements.txt             # Production dependencies
├── start_app.bat                # 1-click launcher for Windows
├── README.md                    # Complete project documentation
│
├── dataset/                     # Training datasets
│   ├── images/                  # Stored image files
│   └── metadata.csv             # Image-caption pair index
│
├── uploads/                     # Artifact storage
│   ├── generated/               # Generated diffusion images
│   └── audio/                   # Generated TTS MP3 files
│
└── templates/
    └── enhanced_index.html      # Multimodal Glassmorphic Web Dashboard
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/harshrameshnerkar/Caption-to-Image.git
cd Caption-to-Image
```

### 2. Setup Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launch Web Studio
```bash
python app.py
```
*(Or double-click `start_app.bat` on Windows)*

Open your browser at **`http://127.0.0.1:5000`**.

---

## 🔌 API Endpoints Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Studio Dashboard |
| `POST` | `/api/generate_image` | Synthesizes image from prompt (`prompt`, `style`, `aspect_ratio`, `seed`) |
| `POST` | `/api/generate_caption` | Generates captions from image (`image` file or base64, `style`) |
| `POST` | `/api/remix` | Multimodal image variation from caption + user steering directives |
| `POST` | `/api/tts` | Generates MP3 audio for caption text |
| `GET` | `/api/dataset/samples` | Retrieves dataset samples and overview statistics |
| `POST` | `/api/dataset/download` | Syncs and downloads seed training dataset pairs |
| `POST` | `/api/train/start` | Initiates background training run (`model_type`, `epochs`, `learning_rate`) |
| `GET` | `/api/train/status` | Returns real-time training step, loss, and epoch progress |
| `POST` | `/api/train/stop` | Stops the active training session |
| `GET` | `/health` | Service health status |

---

## 🧪 Running the Test Suite

Execute the automated end-to-end test suite:
```bash
python test_multimodal.py
```
*Validates Text-to-Image synthesis, Vision-to-Text analysis, Multimodal Remixing, TTS generation, and Dataset Training telemetry.*

---

## 👤 Author

**Harsh Nerkar**
- GitHub: [@harshrameshnerkar](https://github.com/harshrameshnerkar)
- Portfolio: [harshnerkar.vercel.app](https://harshnerkar.vercel.app)

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
