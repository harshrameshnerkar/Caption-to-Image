@echo off
echo Running inference on sample image...
python inference.py --model_path models/best_model.pt --image_path Dataset/Images --style description
echo Inference completed!
pause
