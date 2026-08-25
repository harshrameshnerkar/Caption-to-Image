@echo off
echo Starting model training...
python train.py --epochs 5 --batch_size 8 --learning_rate 5e-5
echo Training completed!
pause
