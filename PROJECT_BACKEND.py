
"""PROJECT _BACKEND.ipynb"""

"""YOLO Object Detection Uses pre-trained YOLOv8m model (80 COCO classes)"""

#  1. Check GPU

!nvidia-smi

#  2. Install dependencies

!pip install ultralytics roboflow fastapi uvicorn nest_asyncio pyngrok opencv-python

from ultralytics import YOLO
import os
import glob
from IPython.display import display, Image
from IPython import display
display.clear_output()


#  3. YOLO environment check

!yolo checks


#  4. Download Pre-trained YOLOv8m Model

print("=" * 60)
print("📥 Downloading Pre-trained YOLOv8m model...")
print("=" * 60)

# This will automatically download yolov8m.pt if not present
model = YOLO('yolov8m.pt')

print("\n✅ Model loaded successfully!")
print("=" * 60)
print("📋 This model can detect 80 COCO classes:")
print("=" * 60)

# Display all classes
for idx, class_name in model.names.items():
    print(f"{idx:2d}: {class_name}")

print("=" * 60)


#  5. Test Model on Sample Images (Optional)

print("\n🧪 Testing model on sample image...")

# Test with a sample image from internet
!wget -q https://ultralytics.com/images/bus.jpg -O test_image.jpg

# Run prediction
results = model.predict('test_image.jpg', conf=0.25, save=True)
print("✅ Test prediction complete! Check results in runs/detect/predict/")

# Display results
from IPython.display import Image as IPImage, display as ipy_display
ipy_display(IPImage(filename='runs/detect/predict/test_image.jpg'))


# Install dependencies (FASTER - only if needed)

import sys
import subprocess

def install_if_needed(packages):
    """Only install if not already installed"""
    for package in packages:
        try:
            __import__(package.split('[')[0])
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install_if_needed(['ultralytics', 'fastapi', 'uvicorn', 'nest_asyncio', 'pyngrok', 'opencv-python'])


#  Imports

import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import threading
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# Apply nest_asyncio for Colab
nest_asyncio.apply()


#  Quick cleanup

os.system("fuser -k 8000/tcp 2>/dev/null")
try:
    ngrok.kill()
except:
    pass


#  Load YOLOv8m (uses cache if available)

print("📦 Loading model...")
model = YOLO("yolov8m.pt")
print(f"✅ Ready! Can detect {len(model.names)} classes\n")


#  FastAPI app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "YOLO Backend is running!", "model": "YOLOv8m", "classes": len(model.names)}

@app.get("/classes")
def get_classes():
    return {"classes": model.names, "total": len(model.names)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)

        results = model.predict(img, conf=0.25, imgsz=640, verbose=False)

        predictions = []
        for r in results:
            for box in r.boxes:
                predictions.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })

        print(f"{'✅' if predictions else '⚠️'} Detected: {len(predictions)} objects")

        return JSONResponse(content={"predictions": predictions, "count": len(predictions)})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Start server

ngrok.set_auth_token("33Jixat1JkYB6pdACXsnUEG3TF3_7XHRemG8JTgJPmNU6buRn")
public_url = ngrok.connect(8000)

print("="*60)
print(f"🚀 URL: {public_url}")
print("="*60)

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

threading.Thread(target=run_server, daemon=True).start()

print("✅ Server running!\n")
print("⚠️ Keep this cell running! Don't stop it or your API will go offline.")
print("💡 You can now use other cells while this runs in background.\n")

# Keep alive (stop with Ctrl+C or Stop button)
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n🛑 Stopping server...")
    ngrok.kill()
    print("✅ Server stopped!")

