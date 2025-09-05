#!/bin/bash
# Setup script for PiCrawler Object Detection

echo "Setting up PiCrawler Object Detection System..."

# Create directories
mkdir -p models
mkdir -p logs

# Install required Python packages
echo "Installing Python dependencies..."
pip3 install opencv-python
pip3 install numpy
pip3 install picamera2
# robot_hat should already be installed from SunFounder setup

# Download YOLOv3-tiny model files (fallback option)
echo "Downloading YOLO model files..."
cd models

# Download YOLOv3-tiny config
if [ ! -f "yolov3-tiny.cfg" ]; then
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
fi

# Download YOLOv3-tiny weights
if [ ! -f "yolov3-tiny.weights" ]; then
    wget https://pjreddie.com/media/files/yolov3-tiny.weights
fi

# Download COCO class names
if [ ! -f "coco.names" ]; then
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
fi

cd ..

echo "Basic setup complete!"
echo ""
echo "IMPORTANT: To run the detection script, you need to activate the virtual environment first:"
echo "  source picrawler_env/bin/activate"
echo "  python picrawler_detection.py"
echo ""
echo "Or create an alias by adding this to ~/.bashrc:"
echo "  alias picrawler='cd $(pwd) && source picrawler_env/bin/activate'"
echo ""
echo "NEXT STEPS:"
echo "1. For Hailo models, download a compatible .hef file to the models/ directory"
echo "2. Check that your camera is working: 'libcamera-hello'"
echo "3. Test robot movement with SunFounder examples"
echo "4. Activate venv and run: 'source picrawler_env/bin/activate && python picrawler_detection.py'"
echo ""
echo "HAILO SETUP (if not done already):"
echo "1. Install Hailo software: https://github.com/hailo-ai/hailo-rpi5-examples"
echo "2. Download pre-trained models from Hailo Model Zoo"
echo "3. Update the model path in the Python script"
