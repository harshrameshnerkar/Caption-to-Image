"""
Test script to demonstrate the enhanced caption generation styles
Tests: Short (1-2 words), Balanced (5-8 lines), Detailed (story format)
"""
import requests
import os
from PIL import Image

def test_caption_generation():
    """Test the enhanced caption generation API"""
    
    print("🧪 Testing Enhanced AI Caption Generation")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get('http://localhost:5000/api/model_status')
        status = response.json()
        
        if not status.get('loaded'):
            print("❌ AI Model not loaded. Please start the server first.")
            return
        
        print("✅ AI Model Status: Ready")
        print(f"📊 PyTorch Version: {status.get('pytorch_version')}")
        print(f"🎨 Available Styles: {status.get('styles_available')}")
        print()
        
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Please start app.py first (python app.py).")
        return
    
    # Find a test image
    test_image_path = None
    image_paths = [
        "Dataset/Images/1000268201_693b08cb0e.jpg",
        "Dataset/Images/1000268201_693b08cb0e.jpg",
    ]
    
    for path in image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if not test_image_path:
        print("❌ No test image found. Please ensure Dataset/Images/ contains images.")
        return
    
    print(f"🖼️ Test Image: {test_image_path}")
    print()
    
    # Test all three styles
    styles = ['short', 'balanced', 'detailed']
    
    for style in styles:
        print(f"🎯 Testing {style.upper()} style:")
        print("-" * 30)
        
        try:
            with open(test_image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {'style': style}
                
                response = requests.post(
                    'http://localhost:5000/api/generate_caption',
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('success'):
                        caption = result.get('caption')
                        word_count = result.get('word_count')
                        
                        print(f"📝 Caption ({style}):")
                        print(f"{caption}")
                        print(f"📊 Word Count: {word_count}")
                        print(f"✅ Status: Success")
                    else:
                        print(f"❌ Error: {result.get('error')}")
                else:
                    print(f"❌ HTTP Error: {response.status_code}")
        
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print()
    
    print("🎉 Caption Generation Test Complete!")
    print("\n📝 Style Descriptions:")
    print("• SHORT: Quick 1-2 word identification")
    print("• BALANCED: Structured 5-8 line detailed description")
    print("• DETAILED: Engaging story-format narrative")
    
    print(f"\n🌐 Web Interface: http://localhost:5000")

if __name__ == "__main__":
    test_caption_generation()