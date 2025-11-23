# Real-Time Facial Recognition & Privacy Application

A web-based application for real-time face detection and privacy protection using OpenCV's DNN module. Deployable to Vercel with browser-based camera access.

## 🌐 Live Demo

[Deploy your own](https://vercel.com/new) or run locally!

## Features

- **Real-time face detection** via webcam
- **Image upload and processing** for static images
- **Multiple privacy filters**:
  - Detection only (bounding boxes)
  - Rectangular blur
  - Elliptical blur
  - Pixelation
  - Combined (elliptical blur + pixelation)
- **Adjustable parameters**:
  - Detection confidence threshold
  - Blur intensity
  - Pixel block size

## 📦 Two Versions Available

### 1. **Web Application** (Recommended for Vercel)
- Flask-based web interface
- Browser camera access
- Works on any device
- See `DEPLOYMENT.md` for Vercel deployment

### 2. **Desktop Application**
- Tkinter GUI
- Direct camera access
- Run locally only
- See `face_recognition_app.py`

## Requirements

### Web Version
- Python 3.7+
- Flask
- opencv-python-headless
- NumPy
- Pillow

### Desktop Version
- Python 3.7+
- OpenCV
- NumPy
- Pillow
- tkinter (usually comes with Python)

## Installation & Usage

### Web Application (For Vercel Deployment)

1. **Prepare models**:
```bash
python prepare_deployment.py
```

2. **Install dependencies**:
```bash
pip install -r requirements_web.txt
```

3. **Run locally**:
```bash
python app.py
```
Visit `http://localhost:5000`

4. **Deploy to Vercel**:
```bash
vercel deploy
```

See `DEPLOYMENT.md` for detailed deployment instructions.

### Desktop Application (Local Only)

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the app**:
```bash
python face_recognition_app.py
```

Note: Desktop app uses models from `/Users/jcors09/Downloads/face-detection-python-master/`

### Camera Mode
1. Click **"Start Camera"** to begin real-time face detection
2. Select a privacy mode from the options
3. Adjust detection and filter parameters using the sliders
4. Click **"Stop Camera"** when done

### Image Mode
1. Click **"Upload Image"** to select an image file
2. The application will detect faces and apply the selected privacy filter
3. Adjust parameters and re-upload to see different effects

## Privacy Modes

### None (Detection Only)
- Draws green bounding boxes around detected faces
- Shows confidence scores
- No privacy filtering applied

### Rectangular Blur
- Applies Gaussian blur to rectangular face regions
- Fast and simple
- May show rectangular artifacts

### Elliptical Blur
- Creates smooth elliptical masks for natural-looking blur
- More aesthetically pleasing than rectangular blur
- Better edge blending

### Pixelate
- Classic pixelation effect
- Downsamples and upsamples face regions
- Configurable pixel block size

### Combined (Ellipse + Pixelate)
- Most sophisticated approach
- Applies both blur and pixelation within elliptical mask
- Best balance of privacy and aesthetics

## Parameters

### Confidence Threshold (0.1 - 1.0)
- Minimum confidence score for face detection
- Higher values = fewer false positives
- Lower values = more detections (may include false positives)
- Default: 0.7

### Blur Factor (1 - 5)
- Controls blur intensity
- Higher values = more blur
- Default: 3

### Pixel Size (5 - 30)
- Size of pixel blocks in pixelation effect
- Higher values = larger blocks (more privacy)
- Default: 16

## Technical Details

### Face Detection Model
- Uses ResNet-10 SSD (Single Shot Detector)
- Pre-trained Caffe model
- Input size: 300×300 pixels
- Mean subtraction: [104, 117, 123]

### Performance
- Real-time processing at ~30 FPS on modern hardware
- Supports multiple face detection in single frame
- Threaded video processing for smooth GUI

## Troubleshooting

### Camera not opening
- Check if another application is using the camera
- Verify camera permissions on macOS/Linux
- Try changing camera index in code (0 to 1, etc.)

### Model not loading
- Ensure model files are in the correct directory
- Check file paths in the code
- Verify model files are not corrupted

### Slow performance
- Reduce camera resolution
- Increase detection threshold
- Close other resource-intensive applications

## Project Structure

```
facial-detection/
├── face_recognition_app.py    # Main application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── model/                      # Model files directory
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
└── FacialRecognition_CV.ipynb # Original notebook
```

## Credits

Based on OpenCV's DNN face detection module and privacy filtering techniques.

## License

MIT License - Feel free to use and modify for your projects.
