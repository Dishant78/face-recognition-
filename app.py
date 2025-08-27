from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

# Import existing modules
from db import init_db, add_person, get_all_persons, delete_person_by_id
from logger import log_detection
from notifier import send_email

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global variables for camera and detection
camera = None
detection_running = False
current_frame = None
detection_results = []

# Initialize database
init_db()

# Load face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
face_cascade_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Load known faces
known_face_metadata = []
known_images = []
known_face_features = []

def create_face_variations(face_image):
    """Create multiple processed versions of a face for better matching"""
    variations = []
    
    # Original resized
    face_resized = cv2.resize(face_image, (100, 100))
    variations.append(face_resized)
    
    # Histogram equalization
    face_eq = cv2.equalizeHist(face_resized)
    variations.append(face_eq)
    
    # Gaussian blur
    face_blur = cv2.GaussianBlur(face_resized, (3, 3), 0)
    variations.append(face_blur)
    
    # Median blur
    face_median = cv2.medianBlur(face_resized, 3)
    variations.append(face_median)
    
    # Bilateral filter
    face_bilateral = cv2.bilateralFilter(face_resized, 9, 75, 75)
    variations.append(face_bilateral)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_clahe = clahe.apply(face_resized)
    variations.append(face_clahe)
    
    # Additional variations for better accuracy
    # Sharpening filter
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    face_sharp = cv2.filter2D(face_resized, -1, kernel)
    variations.append(face_sharp)
    
    # Edge enhancement
    face_edge = cv2.Laplacian(face_resized, cv2.CV_8U)
    variations.append(face_edge)
    
    # Gamma correction
    gamma = 1.2
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    face_gamma = cv2.LUT(face_resized, lookUpTable)
    variations.append(face_gamma)
    
    return variations

def load_known_faces():
    """Load known faces from database"""
    global known_face_metadata, known_images, known_face_features
    
    known_face_metadata = []
    known_images = []
    known_face_features = []
    
    for person in get_all_persons():
        # Updated to use new database schema: (id, name, age, last_seen_location, date_missing, additional_info, contact_email, image_path)
        image_path = person[7]  # image_path is now at index 7
        if os.path.exists(image_path):
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    known_images.append(gray_image)
                    known_face_metadata.append(person)
                    
                    # Create multiple processed versions for better matching
                    processed_versions = create_face_variations(gray_image)
                    known_face_features.append(processed_versions)
                    
                    print(f"✓ Loaded image for {person[1]} (Web UI)")
            except Exception as e:
                print(f"Error processing {person[1]}: {e}")
    
    print(f"Loaded {len(known_images)} known face images for web UI.")

def detect_faces_advanced(gray_frame):
    """Use multiple cascade classifiers for better face detection"""
    faces = []
    
    # Try different cascade classifiers with optimized parameters
    cascades = [
        (face_cascade, 1.05, 4, (40, 40)),      # More sensitive for close faces
        (face_cascade_alt, 1.1, 3, (30, 30)),   # Balanced detection
        (face_cascade_alt2, 1.15, 2, (25, 25))  # More aggressive detection
    ]
    
    for cascade, scale_factor, min_neighbors, min_size in cascades:
        if cascade is not None:
            detected = cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors, 
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces.extend(detected)
    
    # Remove duplicate detections
    if len(faces) > 0:
        faces = np.array(faces)
        # Group nearby detections
        final_faces = []
        for (x, y, w, h) in faces:
            # Check if this detection overlaps significantly with existing ones
            is_duplicate = False
            for (fx, fy, fw, fh) in final_faces:
                # Calculate overlap
                overlap_x = max(0, min(x + w, fx + fw) - max(x, fx))
                overlap_y = max(0, min(y + h, fy + fh) - max(y, fy))
                overlap_area = overlap_x * overlap_y
                min_area = min(w * h, fw * fh)
                
                if overlap_area > 0.6 * min_area:  # 60% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_faces.append((x, y, w, h))
        
        return final_faces
    
    return []

def advanced_template_matching(face_roi, known_face_features):
    """Advanced template matching using multiple variations and methods"""
    best_score = 0
    
    # Try multiple template matching methods
    methods = [
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR_NORMED,
        cv2.TM_SQDIFF_NORMED
    ]
    
    for variation in known_face_features:
        for method in methods:
            # Resize face_roi to match variation size
            face_roi_resized = cv2.resize(face_roi, (100, 100))
            
            result = cv2.matchTemplate(face_roi_resized, variation, method)
            
            if method == cv2.TM_SQDIFF_NORMED:
                # For SQDIFF, lower is better, so invert
                score = 1.0 - np.min(result)
            else:
                score = np.max(result)
            
            best_score = max(best_score, score)
    
    return best_score

def calculate_face_similarity(face1, face2):
    """Calculate similarity between two faces using multiple metrics"""
    # Resize both faces to same size
    face1_resized = cv2.resize(face1, (100, 100))
    face2_resized = cv2.resize(face2, (100, 100))
    
    # Normalize
    face1_norm = cv2.normalize(face1_resized, None, 0, 255, cv2.NORM_MINMAX)
    face2_norm = cv2.normalize(face2_resized, None, 0, 255, cv2.NORM_MINMAX)
    
    # Calculate multiple similarity metrics
    # 1. Template matching
    result = cv2.matchTemplate(face1_norm, face2_norm, cv2.TM_CCOEFF_NORMED)
    template_score = np.max(result)
    
    # 2. Histogram comparison
    hist1 = cv2.calcHist([face1_norm], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2_norm], [0], None, [256], [0, 256])
    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # 3. Mean squared error (inverted and normalized)
    mse = np.mean((face1_norm.astype(float) - face2_norm.astype(float)) ** 2)
    mse_score = 1.0 / (1.0 + mse / 10000)  # Normalize and invert
    
    # 4. Structural similarity (SSIM-like)
    # Calculate local means
    kernel = np.ones((3,3), np.float32) / 9
    mean1 = cv2.filter2D(face1_norm.astype(np.float32), -1, kernel)
    mean2 = cv2.filter2D(face2_norm.astype(np.float32), -1, kernel)
    
    # Calculate local variances
    var1 = cv2.filter2D((face1_norm.astype(np.float32) - mean1)**2, -1, kernel)
    var2 = cv2.filter2D((face2_norm.astype(np.float32) - mean2)**2, -1, kernel)
    
    # Calculate covariance
    covar = cv2.filter2D((face1_norm.astype(np.float32) - mean1) * (face2_norm.astype(np.float32) - mean2), -1, kernel)
    
    # SSIM-like score
    c1, c2 = 0.01, 0.03
    ssim_score = ((2 * np.mean(mean1) * np.mean(mean2) + c1) * (2 * np.mean(covar) + c2)) / \
                ((np.mean(mean1)**2 + np.mean(mean2)**2 + c1) * (np.mean(var1) + np.mean(var2) + c2))
    
    # Combine scores with weights
    combined_score = (0.3 * template_score + 0.2 * hist_score + 0.2 * mse_score + 0.3 * ssim_score)
    return combined_score

def process_frame_for_web(frame):
    """Process frame for web display with detection results"""
    global detection_results
    
    if frame is None:
        return frame, []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use advanced face detection
    faces = detect_faces_advanced(gray)
    
    matches = []
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        best_match = None
        best_score = 0
        
        # Compare with known faces using advanced matching
        for i, known_image in enumerate(known_images):
            if i < len(known_face_features):  # Safety check
                # Use advanced template matching
                score = advanced_template_matching(face_roi, known_face_features[i])
                
                # Also try direct similarity calculation
                similarity_score = calculate_face_similarity(face_roi, known_image)
                
                # Combine both scores with more weight on template matching
                combined_score = (0.6 * score + 0.4 * similarity_score)
                
                # Much stricter threshold for better accuracy (0.75 instead of 0.65)
                if combined_score > 0.75:
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = known_face_metadata[i]
        
        if best_match:
            matches.append((best_match, best_score, (x, y, w, h)))
            # Draw green rectangle around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add text with name and confidence
            cv2.putText(frame, f"{best_match[1]} ({best_score:.2f})", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Draw red rectangle around undetected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Update global detection results
    detection_results = matches
    
    return frame, matches

def generate_frames():
    """Generate camera frames for web streaming"""
    global camera, detection_running, current_frame
    
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            return
        
        detection_running = True
        print("Camera started successfully for web UI")
        
        while detection_running:
            ret, frame = camera.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame for detection
            processed_frame, matches = process_frame_for_web(frame)
            current_frame = processed_frame
            
            # Convert frame to JPEG for web streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        if camera:
            camera.release()
            print("Camera released")

# Routes
@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/add_person', methods=['GET', 'POST'])
def add_person_page():
    """Add missing person page"""
    if request.method == 'POST':
        try:
            # Get form data
            name = request.form['name']
            age = request.form['age']
            last_seen_location = request.form['last_seen_location']
            date_missing = request.form['date_missing']
            additional_info = request.form.get('additional_info', '')
            contact_email = request.form.get('contact_email', '')
            
            # Handle file upload
            if 'photo' in request.files:
                file = request.files['photo']
                if file and file.filename:
                    # Create data directory if it doesn't exist
                    if not os.path.exists('data'):
                        os.makedirs('data')
                    
                    # Save file with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"data/{timestamp}_{file.filename}"
                    file.save(filename)
                    
                    # Add to database
                    add_person(name, age, last_seen_location, date_missing, 
                              additional_info, contact_email, filename)
                    
                    # Reload known faces
                    load_known_faces()
                    
                    return redirect(url_for('home'))
            
            return "Error: No photo uploaded", 400
            
        except Exception as e:
            return f"Error: {str(e)}", 500
    
    return render_template('add_person.html')

@app.route('/camera_feed')
def camera_feed():
    """Camera feed page"""
    return render_template('camera_feed.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_results')
def get_detection_results():
    """Get current detection results for AJAX"""
    global detection_results
    results = []
    for match, confidence, (x, y, w, h) in detection_results:
        results.append({
            'name': match[1],
            'age': match[2],
            'last_seen_location': match[3],
            'date_missing': match[4],
            'additional_info': match[5],
            'confidence': round(confidence, 2),
            'contact_email': match[6]
        })
    return jsonify(results)

@app.route('/view_persons')
def view_persons():
    """View all missing persons"""
    persons = get_all_persons()
    return render_template('view_persons.html', persons=persons)

@app.route('/delete_person/<int:person_id>', methods=['POST'])
def delete_person(person_id):
    """Delete a person from database"""
    try:
        delete_person_by_id(person_id)
        load_known_faces()  # Reload faces
        return redirect(url_for('view_persons'))
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/shutdown')
def shutdown():
    """Shutdown the system"""
    global detection_running, camera
    detection_running = False
    if camera:
        camera.release()
    return "System shutdown successfully"

@app.route('/reload_faces')
def reload_faces():
    """Reload known faces from database"""
    try:
        load_known_faces()
        return jsonify({"status": "success", "message": f"Reloaded {len(known_images)} faces"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Load known faces on startup
print("Loading known faces for web UI...")
load_known_faces()

if __name__ == '__main__':
    print("Starting Flask web application...")
    print("Access the application at: http://localhost:5000")
    print("Improved accuracy settings applied for better detection!")
    app.run(debug=True, host='0.0.0.0', port=5000) 