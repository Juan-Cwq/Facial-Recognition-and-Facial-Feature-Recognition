"""
Web-based Facial Recognition Application for Vercel
Uses Flask for the backend and browser-based camera access
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# Model URLs - hosted externally to avoid deployment size limits
MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

# Model paths in /tmp (Vercel's writable directory)
MODEL_DIR = '/tmp'
MODEL_PATH = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
CONFIG_PATH = os.path.join(MODEL_DIR, 'deploy.prototxt')

def download_models():
    """Download model files if they don't exist"""
    import urllib.request
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print("Downloading model file...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✓ Model downloaded")
    
    if not os.path.exists(CONFIG_PATH):
        print("Downloading config file...")
        urllib.request.urlretrieve(CONFIG_URL, CONFIG_PATH)
        print("✓ Config downloaded")

# Download and load face detection model
net = None
try:
    download_models()
    net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)
    print("✓ Face detection model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    net = None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_faces():
    """Detect faces in uploaded image"""
    try:
        data = request.get_json()
        
        # Get image data and settings
        image_data = data.get('image')
        privacy_mode = data.get('privacy_mode', 'none')
        threshold = float(data.get('threshold', 0.7))
        blur_factor = int(data.get('blur_factor', 3))
        pixel_size = int(data.get('pixel_size', 16))
        
        # Decode base64 image
        if ',' in image_data:
            image_bytes = base64.b64decode(image_data.split(',')[1])
        else:
            image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array directly (faster than PIL)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Failed to decode image'
            }), 400
        
        # Resize for faster processing (optional - improves speed)
        max_width = 1280
        height, width = img.shape[:2]
        if width > max_width:
            scale = max_width / width
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        # Detect faces
        faces = detect_faces_in_image(img, threshold)
        
        # Apply privacy filter
        processed = apply_privacy_filter(img, faces, privacy_mode, blur_factor, pixel_size)
        
        # Convert back to base64 (faster encoding)
        _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{processed_base64}',
            'faces_count': len(faces)
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def detect_faces_in_image(image, threshold=0.7):
    """Detect faces in an image"""
    if net is None:
        return []
        
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    h, w = image.shape[:2]
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2, y2, confidence))
            
    return faces

def blur_face(face, factor=3):
    """Apply Gaussian blur to face region"""
    h, w = face.shape[:2]
    factor = max(1, min(5, factor))
    
    w_k = int(w / factor)
    h_k = int(h / factor)
    
    if w_k % 2 == 0:
        w_k += 1
    if h_k % 2 == 0:
        h_k += 1
        
    blurred = cv2.GaussianBlur(face, (w_k, h_k), 0, 0)
    return blurred

def pixelate_face(face, pixels=16):
    """Pixelate face region"""
    h, w = face.shape[:2]
    
    if h > pixels and w > pixels:
        face_small = cv2.resize(face, (pixels, pixels), interpolation=cv2.INTER_LINEAR)
        face_pixelated = cv2.resize(face_small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        face_pixelated = face
        
    return face_pixelated

def apply_privacy_filter(image, faces, mode='none', blur_factor=3, pixel_size=16):
    """Apply selected privacy filter to detected faces"""
    img = image.copy()
    
    if mode == 'none':
        # Just draw bounding boxes
        for (x1, y1, x2, y2, conf) in faces:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Face: {conf:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    elif mode == 'blur':
        # Rectangular blur
        for (x1, y1, x2, y2, conf) in faces:
            face = img[y1:y2, x1:x2]
            face = blur_face(face, blur_factor)
            img[y1:y2, x1:x2] = face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    elif mode == 'ellipse':
        # Elliptical blur
        img_blur = img.copy()
        elliptical_mask = np.zeros(img.shape, dtype=img.dtype)
        
        for (x1, y1, x2, y2, conf) in faces:
            face = img[y1:y2, x1:x2]
            face = blur_face(face, blur_factor)
            img_blur[y1:y2, x1:x2] = face
            
            e_center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
            e_size = (int(x2 - x1), int(y2 - y1))
            e_angle = 0.0
            
            cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle), (255, 255, 255), -1, cv2.LINE_AA)
            
        np.putmask(img, elliptical_mask, img_blur)
        
    elif mode == 'pixelate':
        # Pixelate
        for (x1, y1, x2, y2, conf) in faces:
            face = img[y1:y2, x1:x2]
            face = pixelate_face(face, pixel_size)
            img[y1:y2, x1:x2] = face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    elif mode == 'combined':
        # Combined elliptical blur + pixelate
        img_out = img.copy()
        elliptical_mask = np.zeros(img.shape, dtype=img.dtype)
        
        for (x1, y1, x2, y2, conf) in faces:
            face = img[y1:y2, x1:x2]
            face = blur_face(face, blur_factor)
            face = pixelate_face(face, pixel_size)
            img_out[y1:y2, x1:x2] = face
            
            e_center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
            e_size = (int(x2 - x1), int(y2 - y1))
            e_angle = 0.0
            
            cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle), (255, 255, 255), -1, cv2.LINE_AA)
            
        np.putmask(img, elliptical_mask, img_out)
        
    return img

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
