#!/bin/bash

# 1. Add FastSAM directly from the CASIA-LMC-Lab repository
# This installs the 'fastsam' package so you can import it in your code.
uv add "fastsam @ git+https://github.com/CASIA-LMC-Lab/FastSAM.git"

# 2. Add CLIP (Required for text prompt mode)
uv add "clip @ git+https://github.com/openai/CLIP.git"

# 3. Add Core Dependencies (based on FastSAM requirements)
# Note: FastSAM is built on Ultralytics YOLOv8.
uv add ultralytics numpy matplotlib opencv-python torch torchvision

# 4. (Optional) Add Gradio if you plan to run the web demo (app_gradio.py)
uv add gradio

echo "FastSAM and dependencies installed successfully via uv!"