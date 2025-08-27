print("main.py started - BROWSER-SPECIFIC OPTIMIZATION")

try:
    from db import init_db, get_all_persons
    from camera import start_camera
    from logger import log_detection
    from notifier import send_email
    import cv2
    import numpy as np
    import os
    import time
    import threading
    from collections import deque
    import gc  # Garbage collection for browser memory management
    
    # Try to import face_recognition, but don't fail if it doesn't work
    try:
        import face_recognition
        USE_FACE_RECOGNITION = True
        print("✓ face_recognition library loaded successfully")
    except ImportError:
        USE_FACE_RECOGNITION = False
        print("✗ face_recognition library not available, using OpenCV fallback")
    
    print("Inside main block - Browser Optimization Mode")
    
    # Initialize database
    print("Initializing DB...")
    init_db()
    
    # BROWSER-SPECIFIC PERFORMANCE CONSTANTS
    class BrowserConfig:
        # Ultra-conservative settings for browsers
        PROCESS_EVERY_N_FRAMES = 20      # Process every 20th frame (very conservative)
        DETECTION_COOLDOWN = 3.0         # 3 second cooldown for browsers
        MAX_FRAME_WIDTH = 320            # Much smaller for browser performance
        MAX_FRAME_HEIGHT = 240           # Limit height too
        FACE_MIN_SIZE = (25, 25)         # Slightly smaller minimum face
        
        # Face recognition settings (more lenient for browser accuracy)
        FACE_RECOGNITION_TOLERANCE = 0.65    # More lenient for browser
        OPENCV_SIMILARITY_THRESHOLD = 0.45   # Lower threshold for browser detection
        
        # Memory management
        MAX_CACHE_SIZE = 3               # Smaller cache for browsers
        GARBAGE_COLLECT_INTERVAL = 50    # Clean memory every 50 frames
        
        # Processing optimization
        USE_SINGLE_CASCADE_ONLY = True   # Use only one cascade for speed
        SKIP_HISTOGRAM_COMPARISON = False # Keep histogram but make it optional
        REDUCE_FACE_PREPROCESSING = True # Minimal preprocessing
    
    # Global variables optimized for browsers
    frame_count = 0
    last_detection_time = 0
    last_faces = []
    processing = False
    detection_queue = deque(maxlen=BrowserConfig.MAX_CACHE_SIZE)
    last_cleanup_time = 0
    
    # Threading lock
    processing_lock = threading.Lock()
    
    def cleanup_memory():
        """Force garbage collection for browser memory management"""
        global last_cleanup_time
        current_time = time.time()
        
        if current_time - last_cleanup_time > 10:  # Every 10 seconds
            gc.collect()
            last_cleanup_time = current_time
            print("🧹 Memory cleanup performed")
    
    def load_single_best_cascade():
        """Load only the best performing cascade for browsers"""
        cascade_priority = [
            'haarcascade_frontalface_default.xml',
            'haarcascade_frontalface_alt.xml',
            'haarcascade_frontalface_alt2.xml'
        ]
        
        for cascade_file in cascade_priority:
            try:
                cascade_path = cv2.data.haarcascades + cascade_file
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    print(f"✓ Loaded {cascade_file} for browser optimization")
                    return cascade
            except Exception as e:
                print(f"✗ Failed to load {cascade_file}: {e}")
                continue
        
        print("✗ No face cascades could be loaded!")
        return None
    
    # Load single cascade for browser performance
    face_cascade = load_single_best_cascade()
    
    def browser_optimize_frame(frame):
        """Ultra-aggressive frame optimization for browsers"""
        if frame is None:
            return None, 1.0
            
        height, width = frame.shape[:2]
        
        # Calculate scale to fit within browser limits
        width_scale = BrowserConfig.MAX_FRAME_WIDTH / width if width > BrowserConfig.MAX_FRAME_WIDTH else 1.0
        height_scale = BrowserConfig.MAX_FRAME_HEIGHT / height if height > BrowserConfig.MAX_FRAME_HEIGHT else 1.0
        scale = min(width_scale, height_scale)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Use fastest interpolation for browsers
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            return resized_frame, scale
        
        return frame, 1.0
    
    def extract_face_browser_optimized(image_path):
        """Ultra-fast face extraction for browser compatibility"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Aggressive resizing for speed
            height, width = image.shape[:2]
            if width > 200:  # Even smaller for database images
                scale = 200.0 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if face_cascade is None:
                # If no cascade, return resized image
                return cv2.resize(gray, (48, 48))  # Even smaller
            
            # Ultra-fast detection
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,  # Faster but less accurate
                minNeighbors=2,   # Lower for speed
                minSize=(15, 15), # Smaller minimum
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # Get largest face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                x, y, w, h = largest_face
                
                # Extract with no padding for speed
                face_roi = gray[y:y+h, x:x+w]
                return cv2.resize(face_roi, (48, 48))  # Smaller size
            else:
                return cv2.resize(gray, (48, 48))
                
        except Exception as e:
            print(f"Browser face extraction error for {image_path}: {e}")
            return None
    
    def preprocess_face_minimal(face_image):
        """Minimal face preprocessing for browser speed"""
        if face_image is None:
            return None
        
        # Just resize, skip other preprocessing if configured
        if BrowserConfig.REDUCE_FACE_PREPROCESSING:
            return cv2.resize(face_image, (48, 48))
        
        # Minimal processing
        face_resized = cv2.resize(face_image, (48, 48))
        return cv2.equalizeHist(face_resized)
    
    def detect_faces_browser_fast(gray_frame):
        """Ultra-fast face detection for browsers"""
        if face_cascade is None:
            return []
        
        # Ultra-conservative detection for browser performance
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=BrowserConfig.FACE_MIN_SIZE,
            maxSize=(150, 150),  # Limit max size for performance
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Limit number of faces processed
        faces_list = faces.tolist() if len(faces) > 0 else []
        return faces_list[:3]  # Process max 3 faces for browser performance
    
    def compare_faces_browser_optimized(face1, face2):
        """Browser-optimized face comparison with fallbacks"""
        if face1 is None or face2 is None:
            return 0.0
        
        try:
            # Preprocess faces
            norm_face1 = preprocess_face_minimal(face1)
            norm_face2 = preprocess_face_minimal(face2)
            
            if norm_face1 is None or norm_face2 is None:
                return 0.0
            
            # Primary method: Template matching (fast)
            result = cv2.matchTemplate(norm_face1, norm_face2, cv2.TM_CCOEFF_NORMED)
            template_score = np.max(result)
            
            # Secondary method: Histogram (optional for browsers)
            if not BrowserConfig.SKIP_HISTOGRAM_COMPARISON:
                try:
                    hist1 = cv2.calcHist([norm_face1], [0], None, [64], [0, 256])  # Fewer bins
                    hist2 = cv2.calcHist([norm_face2], [0], None, [64], [0, 256])  # Fewer bins
                    hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    
                    # Weighted combination
                    final_score = (template_score * 0.8) + (hist_score * 0.2)
                    return max(0.0, final_score)
                except:
                    return max(0.0, template_score)
            else:
                return max(0.0, template_score)
            
        except Exception as e:
            return 0.0
    
    def draw_browser_optimized(frame, x, y, w, h, person_name=None, confidence=None):
        """Ultra-minimal drawing for browser performance"""
        try:
            if person_name and confidence:
                # Simple green rectangle for found person
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Minimal text
                text = f"{person_name}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Simple red rectangle for unknown
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        except Exception as e:
            # If drawing fails, just skip it
            pass
    
    # Load known faces with browser-optimized extraction
    print("Loading known faces with BROWSER optimization...")
    known_face_encodings = []
    known_face_metadata = []
    known_face_images = []
    
    persons = get_all_persons()
    total_persons = len(persons)
    loaded_count = 0
    
    print(f"Processing {total_persons} persons for browser compatibility...")
    
    for i, person in enumerate(persons):
        if i % 10 == 0:  # Progress indicator
            print(f"Progress: {i}/{total_persons}")
            
        image_path = person[7]
        if not os.path.exists(image_path):
            continue
        
        try:
            # For browsers, prioritize OpenCV method for compatibility
            extracted_face = extract_face_browser_optimized(image_path)
            if extracted_face is not None:
                known_face_images.append(extracted_face)
                known_face_metadata.append(person)
                loaded_count += 1
            
            # Only try face_recognition as fallback and if specifically enabled
            if USE_FACE_RECOGNITION and extracted_face is None:
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image, num_jitters=1)
                    
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_metadata.append(person)
                        loaded_count += 1
                        
                except Exception:
                    pass  # Silent fail for browser compatibility
                    
        except Exception as e:
            print(f"Error processing {person[1]}: {e}")
    
    print(f"Browser loading complete: {loaded_count}/{total_persons} persons loaded")
    print(f"OpenCV faces: {len(known_face_images)}")
    print(f"Face recognition encodings: {len(known_face_encodings)}")
    
    def process_frame_browser_optimized(frame):
        """ULTRA-OPTIMIZED frame processing specifically for browsers"""
        global frame_count, last_detection_time, last_faces, processing
        
        if frame is None:
            return []
        
        current_time = time.time()
        frame_count += 1
        
        # Memory cleanup
        if frame_count % BrowserConfig.GARBAGE_COLLECT_INTERVAL == 0:
            cleanup_memory()
        
        # Thread safety
        if processing:
            # Draw cached results immediately
            for face_info in last_faces:
                draw_browser_optimized(
                    frame, 
                    face_info['x'], face_info['y'], face_info['w'], face_info['h'],
                    face_info.get('name'), face_info.get('confidence')
                )
            return []
        
        # Check if we should process (very conservative for browsers)
        should_process = (
            (frame_count % BrowserConfig.PROCESS_EVERY_N_FRAMES == 0) and
            (current_time - last_detection_time > BrowserConfig.DETECTION_COOLDOWN)
        )
        
        if not should_process:
            # Draw cached results
            for face_info in last_faces:
                draw_browser_optimized(
                    frame, 
                    face_info['x'], face_info['y'], face_info['w'], face_info['h'],
                    face_info.get('name'), face_info.get('confidence')
                )
            return []
        
        processing = True
        matches = []
        current_faces = []
        
        try:
            # Ultra-aggressive frame optimization
            optimized_frame, scale = browser_optimize_frame(frame)
            
            if optimized_frame is None:
                return []
            
            # Prioritize OpenCV method for browser compatibility
            if len(known_face_images) > 0:
                gray_frame = cv2.cvtColor(optimized_frame, cv2.COLOR_BGR2GRAY)
                detected_faces = detect_faces_browser_fast(gray_frame)
                
                print(f"Browser detected {len(detected_faces)} faces")
                
                for face_idx, (x, y, w, h) in enumerate(detected_faces):
                    if face_idx >= 2:  # Limit to 2 faces for browser performance
                        break
                        
                    # Scale coordinates back
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(w / scale)
                    orig_h = int(h / scale)
                    
                    # Skip tiny faces
                    if orig_w < 20 or orig_h < 20:
                        continue
                    
                    # Extract face from original frame
                    try:
                        gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        face_roi = gray_original[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w]
                        
                        if face_roi.size == 0:
                            continue
                        
                        # Find best match with lower threshold for browsers
                        best_match = None
                        best_score = 0.0
                        
                        for idx, known_face in enumerate(known_face_images[:min(len(known_face_images), 20)]):  # Limit comparisons
                            if idx >= len(known_face_metadata):
                                break
                            
                            similarity_score = compare_faces_browser_optimized(face_roi, known_face)
                            
                            if similarity_score > best_score:
                                best_score = similarity_score
                                best_match = known_face_metadata[idx]
                        
                        if best_match and best_score > BrowserConfig.OPENCV_SIMILARITY_THRESHOLD:
                            # Found match
                            draw_browser_optimized(frame, orig_x, orig_y, orig_w, orig_h, best_match[1], best_score)
                            
                            current_faces.append({
                                'x': orig_x, 'y': orig_y, 'w': orig_w, 'h': orig_h,
                                'name': best_match[1], 'confidence': best_score
                            })
                            matches.append((best_match, best_score))
                            print(f"🎯 BROWSER DETECTION: {best_match[1]} (Score: {best_score:.3f})")
                        else:
                            # Unknown face
                            draw_browser_optimized(frame, orig_x, orig_y, orig_w, orig_h)
                            current_faces.append({
                                'x': orig_x, 'y': orig_y, 'w': orig_w, 'h': orig_h
                            })
                            
                    except Exception as face_error:
                        print(f"Face processing error: {face_error}")
                        continue
            
            # Try face_recognition only as backup and if available
            elif USE_FACE_RECOGNITION and len(known_face_encodings) > 0:
                try:
                    rgb_frame = cv2.cvtColor(optimized_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        for (top, right, bottom, left), face_encoding in zip(face_locations[:2], face_encodings[:2]):
                            orig_x = int(left / scale)
                            orig_y = int(top / scale)
                            orig_w = int((right - left) / scale)
                            orig_h = int((bottom - top) / scale)
                            
                            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(distances)
                            
                            if distances[best_match_index] < BrowserConfig.FACE_RECOGNITION_TOLERANCE:
                                person_name = known_face_metadata[best_match_index][1]
                                confidence = 1.0 - distances[best_match_index]
                                
                                draw_browser_optimized(frame, orig_x, orig_y, orig_w, orig_h, person_name, confidence)
                                
                                current_faces.append({
                                    'x': orig_x, 'y': orig_y, 'w': orig_w, 'h': orig_h,
                                    'name': person_name, 'confidence': confidence
                                })
                                matches.append((known_face_metadata[best_match_index], confidence))
                            else:
                                draw_browser_optimized(frame, orig_x, orig_y, orig_w, orig_h)
                                current_faces.append({
                                    'x': orig_x, 'y': orig_y, 'w': orig_w, 'h': orig_h
                                })
                                
                except Exception as fr_error:
                    print(f"Face recognition error: {fr_error}")
            
            # Update cache
            last_faces = current_faces
            last_detection_time = current_time
            
            # Log matches
            for match_data, confidence in matches:
                try:
                    log_detection(match_data[1], match_data[7])
                    print(f"🚨 BROWSER ALERT: {match_data[1]} detected (Confidence: {confidence:.3f})")
                except Exception as e:
                    print(f"Logging error: {e}")
            
        except Exception as e:
            print(f"Browser frame processing error: {e}")
        finally:
            processing = False
        
        return [match for match, _ in matches]
    
    # Print browser-specific performance info
    def print_browser_performance_info():
        print("\n" + "="*60)
        print("🌐 BROWSER-SPECIFIC MISSING PERSON DETECTION")
        print("="*60)
        print(f"🔧 Browser Optimizations:")
        print(f"   • Process every {BrowserConfig.PROCESS_EVERY_N_FRAMES} frames")
        print(f"   • Detection cooldown: {BrowserConfig.DETECTION_COOLDOWN}s")
        print(f"   • Max frame: {BrowserConfig.MAX_FRAME_WIDTH}x{BrowserConfig.MAX_FRAME_HEIGHT}")
        print(f"   • Face size: 48x48 pixels")
        print(f"   • Memory cleanup every {BrowserConfig.GARBAGE_COLLECT_INTERVAL} frames")
        print(f"📊 Database Status:")
        print(f"   • OpenCV faces: {len(known_face_images)}")
        print(f"   • Face encodings: {len(known_face_encodings)}")
        print(f"   • Total loaded: {len(known_face_images) + len(known_face_encodings)}")
        print(f"🎯 Detection Thresholds:")
        print(f"   • OpenCV threshold: {BrowserConfig.OPENCV_SIMILARITY_THRESHOLD}")
        print(f"   • Face recognition tolerance: {BrowserConfig.FACE_RECOGNITION_TOLERANCE}")
        print("="*60)
        print("🚀 OPTIMIZED FOR BROWSER PERFORMANCE!")
        print("Press 'q' to quit")
        print("="*60 + "\n")
    
    # Start the browser-optimized system
    print_browser_performance_info()
    start_camera(process_frame_browser_optimized)
    
except Exception as e:
    print(f"CRITICAL BROWSER ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*50)
    print("BROWSER SYSTEM FAILED - Check error above")
    print("="*50)