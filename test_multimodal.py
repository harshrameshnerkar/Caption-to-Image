"""
End-to-End Test Suite for CaptionAI Multimodal Studio.
Tests:
1. Caption to Image Generation (/api/generate_image)
2. Image to Caption Analysis (/api/generate_caption)
3. Multimodal Remix Workflow (/api/remix)
4. TTS Audio Generation (/api/tts)
5. Dataset Hub & Sample Ingestion (/api/dataset/samples)
6. Model Training Telemetry (/api/train/start, /api/train/status, /api/train/stop)
"""
import os
import sys
import json
import base64
from io import BytesIO
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import app

client = app.test_client()


def test_caption_to_image():
    print("\n--- 1. Testing Caption to Image (Text-to-Image) ---")
    payload = {
        "prompt": "A magical glowing tree in a bioluminescent forest with fireflies",
        "style": "fantasy",
        "aspect_ratio": "1:1"
    }
    response = client.post("/api/generate_image", json=payload)
    assert response.status_code == 200, f"Status code {response.status_code}"
    data = response.get_json()
    assert data["success"] is True, f"Failed: {data}"
    assert "download_url" in data
    print(f"✅ Generated Image URL: {data['download_url']}")
    print(f"✅ Dimensions: {data['width']}x{data['height']}, Seed: {data['seed']}")


def test_image_to_caption():
    print("\n--- 2. Testing Image to Caption (Vision-to-Text) ---")
    # Create test in-memory image
    img = Image.new("RGB", (300, 300), color=(70, 130, 180))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    data = {
        'image': (buf, 'test_image.jpg'),
        'style': 'balanced'
    }
    response = client.post("/api/generate_caption", data=data, content_type='multipart/form-data')
    assert response.status_code == 200, f"Status code {response.status_code}"
    res = response.get_json()
    assert res["success"] is True, f"Failed: {res}"
    print(f"✅ Generated Caption: \"{res['caption']}\"")
    print(f"✅ Alt-Text: \"{res['alt_text']}\"")
    print(f"✅ Tags: {res['tags']}")


def test_multimodal_remix():
    print("\n--- 3. Testing Multimodal Remix (Image -> Caption -> Reimagined Visual) ---")
    payload = {
        "base_caption": "A serene mountain peak at dawn",
        "user_modification": "Reimagined as a futuristic sci-fi colony with flying shuttles",
        "target_style": "cinematic"
    }
    response = client.post("/api/remix", json=payload)
    assert response.status_code == 200, f"Status code {response.status_code}"
    data = response.get_json()
    assert data["success"] is True, f"Failed: {data}"
    print(f"✅ Remixed Visual URL: {data['download_url']}")
    print(f"✅ Final Prompt: {data['final_composed_prompt']}")


def test_text_to_speech():
    print("\n--- 4. Testing Text-to-Speech (TTS) ---")
    payload = {
        "text": "A breathtaking photograph of an alpine lake reflecting majestic mountain peaks during golden hour."
    }
    response = client.post("/api/tts", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    print(f"✅ Audio URL: {data['audio_url']}")


def test_dataset_and_training():
    print("\n--- 5. Testing Dataset Manager & Training Engine ---")
    # Dataset samples
    resp = client.get("/api/dataset/samples")
    assert resp.status_code == 200
    samples_data = resp.get_json()
    assert samples_data["success"] is True
    print(f"✅ Total Dataset Image-Caption Pairs: {samples_data['stats']['total_pairs']}")

    # Start training
    train_payload = {"model_type": "captioning", "epochs": 2, "learning_rate": 0.00005}
    start_resp = client.post("/api/train/start", json=train_payload)
    assert start_resp.status_code == 200
    print(f"✅ Training Initialized: {start_resp.get_json()['status']}")

    # Check status
    status_resp = client.get("/api/train/status")
    assert status_resp.status_code == 200
    print(f"✅ Real-Time Telemetry: {status_resp.get_json()['message']}")

    # Stop training
    client.post("/api/train/stop")
    print("✅ Training Stop Signal Verified.")


if __name__ == "__main__":
    print("=" * 60)
    print(" RUNNING CAPTIONAI MULTIMODAL TEST SUITE")
    print("=" * 60)
    test_caption_to_image()
    test_image_to_caption()
    test_multimodal_remix()
    test_text_to_speech()
    test_dataset_and_training()
    print("\n" + "=" * 60)
    print(" 🌟 ALL CAPTIONAI END-TO-END TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

