"""
Real-Time Facial Recognition Application
Features:
- Real-time face detection via webcam
- Upload and process images
- Multiple privacy filters (blur, pixelate, elliptical)
- Face tracking and bounding boxes
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition & Privacy App")
        self.root.geometry("1200x800")
        
        # Model paths - using files from Downloads folder
        model_dir = '/Users/jcors09/Downloads/face-detection-python-master'
        self.model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        self.config_path = os.path.join(model_dir, 'deploy.prototxt.txt')
        
        # Load face detection model
        try:
            self.net = cv2.dnn.readNetFromCaffe(self.config_path, self.model_path)
            print("✓ Face detection model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}\nPlease ensure model files are in the 'model' directory")
            self.net = None
        
        # Video capture
        self.cap = None
        self.is_camera_running = False
        self.current_frame = None
        self.processed_frame = None
        
        # Settings
        self.detection_threshold = 0.7
        self.blur_factor = 3
        self.pixel_size = 16
        self.privacy_mode = "none"  # none, blur, ellipse, pixelate, combined
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Create the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera controls
        ttk.Label(control_frame, text="Camera Controls", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        self.btn_start_camera = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.btn_start_camera.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.btn_stop_camera = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state='disabled')
        self.btn_stop_camera.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        # Image upload
        ttk.Label(control_frame, text="Image Processing", font=('Arial', 12, 'bold')).grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Button(control_frame, text="Upload Image", command=self.upload_image).grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        # Privacy mode selection
        ttk.Label(control_frame, text="Privacy Mode", font=('Arial', 12, 'bold')).grid(row=7, column=0, columnspan=2, pady=(0, 10))
        
        self.privacy_var = tk.StringVar(value="none")
        privacy_modes = [
            ("None (Detection Only)", "none"),
            ("Rectangular Blur", "blur"),
            ("Elliptical Blur", "ellipse"),
            ("Pixelate", "pixelate"),
            ("Combined (Ellipse + Pixelate)", "combined")
        ]
        
        for i, (text, mode) in enumerate(privacy_modes):
            ttk.Radiobutton(control_frame, text=text, variable=self.privacy_var, 
                          value=mode, command=self.update_privacy_mode).grid(row=8+i, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        ttk.Separator(control_frame, orient='horizontal').grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)
        
        # Detection settings
        ttk.Label(control_frame, text="Detection Settings", font=('Arial', 12, 'bold')).grid(row=14, column=0, columnspan=2, pady=(0, 10))
        
        ttk.Label(control_frame, text="Confidence Threshold:").grid(row=15, column=0, sticky=tk.W)
        self.threshold_var = tk.DoubleVar(value=0.7)
        threshold_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.threshold_var, 
                                   orient='horizontal', command=self.update_threshold)
        threshold_scale.grid(row=16, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.threshold_label = ttk.Label(control_frame, text="0.70")
        self.threshold_label.grid(row=17, column=0, columnspan=2)
        
        ttk.Label(control_frame, text="Blur Factor:").grid(row=18, column=0, sticky=tk.W, pady=(10, 0))
        self.blur_var = tk.IntVar(value=3)
        blur_scale = ttk.Scale(control_frame, from_=1, to=5, variable=self.blur_var, 
                              orient='horizontal', command=self.update_blur)
        blur_scale.grid(row=19, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.blur_label = ttk.Label(control_frame, text="3")
        self.blur_label.grid(row=20, column=0, columnspan=2)
        
        ttk.Label(control_frame, text="Pixel Size:").grid(row=21, column=0, sticky=tk.W, pady=(10, 0))
        self.pixel_var = tk.IntVar(value=16)
        pixel_scale = ttk.Scale(control_frame, from_=5, to=30, variable=self.pixel_var, 
                               orient='horizontal', command=self.update_pixel)
        pixel_scale.grid(row=22, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.pixel_label = ttk.Label(control_frame, text="16")
        self.pixel_label.grid(row=23, column=0, columnspan=2)
        
        # Right panel - Video display
        display_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        display_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(display_frame)
        self.video_label.pack(expand=True)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        
        self.faces_label = ttk.Label(status_frame, text="Faces detected: 0", anchor=tk.E)
        self.faces_label.pack(fill=tk.X)
        
    def update_privacy_mode(self):
        """Update privacy mode setting"""
        self.privacy_mode = self.privacy_var.get()
        
    def update_threshold(self, value):
        """Update detection threshold"""
        self.detection_threshold = float(value)
        self.threshold_label.config(text=f"{self.detection_threshold:.2f}")
        
    def update_blur(self, value):
        """Update blur factor"""
        self.blur_factor = int(float(value))
        self.blur_label.config(text=str(self.blur_factor))
        
    def update_pixel(self, value):
        """Update pixel size"""
        self.pixel_size = int(float(value))
        self.pixel_label.config(text=str(self.pixel_size))
        
    def start_camera(self):
        """Start the camera feed"""
        if self.net is None:
            messagebox.showerror("Error", "Face detection model not loaded")
            return
            
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
            
        self.is_camera_running = True
        self.btn_start_camera.config(state='disabled')
        self.btn_stop_camera.config(state='normal')
        self.status_label.config(text="Camera running...")
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.update_frame, daemon=True)
        self.video_thread.start()
        
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.btn_start_camera.config(state='normal')
        self.btn_stop_camera.config(state='disabled')
        self.status_label.config(text="Camera stopped")
        
    def detect_faces(self, image):
        """Detect faces in an image"""
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        h, w = image.shape[:2]
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence >= self.detection_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2, y2, confidence))
                
        return faces
    
    def blur_face(self, face, factor=3):
        """Apply Gaussian blur to face region"""
        h, w = face.shape[:2]
        if factor < 1:
            factor = 1
        if factor > 5:
            factor = 5
            
        w_k = int(w / factor)
        h_k = int(h / factor)
        
        if w_k % 2 == 0:
            w_k += 1
        if h_k % 2 == 0:
            h_k += 1
            
        blurred = cv2.GaussianBlur(face, (w_k, h_k), 0, 0)
        return blurred
    
    def pixelate_face(self, face, pixels=16):
        """Pixelate face region"""
        h, w = face.shape[:2]
        
        if h > pixels and w > pixels:
            face_small = cv2.resize(face, (pixels, pixels), interpolation=cv2.INTER_LINEAR)
            face_pixelated = cv2.resize(face_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            face_pixelated = face
            
        return face_pixelated
    
    def apply_privacy_filter(self, image, faces):
        """Apply selected privacy filter to detected faces"""
        img = image.copy()
        
        if self.privacy_mode == "none":
            # Just draw bounding boxes
            for (x1, y1, x2, y2, conf) in faces:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Face: {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        elif self.privacy_mode == "blur":
            # Rectangular blur
            for (x1, y1, x2, y2, conf) in faces:
                face = img[y1:y2, x1:x2]
                face = self.blur_face(face, self.blur_factor)
                img[y1:y2, x1:x2] = face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        elif self.privacy_mode == "ellipse":
            # Elliptical blur
            img_blur = img.copy()
            elliptical_mask = np.zeros(img.shape, dtype=img.dtype)
            
            for (x1, y1, x2, y2, conf) in faces:
                face = img[y1:y2, x1:x2]
                face = self.blur_face(face, self.blur_factor)
                img_blur[y1:y2, x1:x2] = face
                
                e_center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                e_size = (int(x2 - x1), int(y2 - y1))
                e_angle = 0.0
                
                cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle), (255, 255, 255), -1, cv2.LINE_AA)
                
            np.putmask(img, elliptical_mask, img_blur)
            
        elif self.privacy_mode == "pixelate":
            # Pixelate
            for (x1, y1, x2, y2, conf) in faces:
                face = img[y1:y2, x1:x2]
                face = self.pixelate_face(face, self.pixel_size)
                img[y1:y2, x1:x2] = face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
        elif self.privacy_mode == "combined":
            # Combined elliptical blur + pixelate
            img_out = img.copy()
            elliptical_mask = np.zeros(img.shape, dtype=img.dtype)
            
            for (x1, y1, x2, y2, conf) in faces:
                face = img[y1:y2, x1:x2]
                face = self.blur_face(face, self.blur_factor)
                face = self.pixelate_face(face, self.pixel_size)
                img_out[y1:y2, x1:x2] = face
                
                e_center = (int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                e_size = (int(x2 - x1), int(y2 - y1))
                e_angle = 0.0
                
                cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle), (255, 255, 255), -1, cv2.LINE_AA)
                
            np.putmask(img, elliptical_mask, img_out)
            
        return img
    
    def update_frame(self):
        """Update video frame continuously"""
        while self.is_camera_running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Apply privacy filter
            processed = self.apply_privacy_filter(frame, faces)
            
            # Convert to RGB for display
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            display_height = 600
            aspect_ratio = processed_rgb.shape[1] / processed_rgb.shape[0]
            display_width = int(display_height * aspect_ratio)
            processed_rgb = cv2.resize(processed_rgb, (display_width, display_height))
            
            # Convert to PhotoImage
            img_pil = Image.fromarray(processed_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            # Update label
            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)
            
            # Update face count
            self.faces_label.config(text=f"Faces detected: {len(faces)}")
            
    def upload_image(self):
        """Upload and process an image"""
        if self.net is None:
            messagebox.showerror("Error", "Face detection model not loaded")
            return
            
        filename = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        # Load image
        img = cv2.imread(filename)
        if img is None:
            messagebox.showerror("Error", "Could not load image")
            return
            
        # Detect faces
        faces = self.detect_faces(img)
        
        # Apply privacy filter
        processed = self.apply_privacy_filter(img, faces)
        
        # Convert to RGB for display
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        display_height = 600
        aspect_ratio = processed_rgb.shape[1] / processed_rgb.shape[0]
        display_width = int(display_height * aspect_ratio)
        processed_rgb = cv2.resize(processed_rgb, (display_width, display_height))
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(processed_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        # Update label
        self.video_label.img_tk = img_tk
        self.video_label.config(image=img_tk)
        
        # Update status
        self.status_label.config(text=f"Processed: {os.path.basename(filename)}")
        self.faces_label.config(text=f"Faces detected: {len(faces)}")
        
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
