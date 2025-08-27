import face_recognition
import cv2
import numpy as np
import os
from db import get_all_persons

def load_known_faces():
    """Load known faces from database with improved error handling"""
    known_encodings = []
    known_metadata = []
    
    for person in get_all_persons():
        image_path = person[6]
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        try:
            # Load image using PIL first to ensure proper format
            from PIL import Image
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Save as temporary clean image
            temp_path = f"temp_{person[0]}.jpg"
            pil_image.save(temp_path, 'JPEG', quality=95)
            
            # Load with face_recognition
            image = face_recognition.load_image_file(temp_path)
            print(f"Loaded image for {person[1]}: shape={image.shape}, dtype={image.dtype}")
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])  # Use first face found
                known_metadata.append(person)
                print(f"Successfully encoded face for {person[1]}")
            else:
                print(f"No face found in image for {person[1]}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Error processing {person[1]}: {e}")
            continue
    
    return known_encodings, known_metadata

def recognize_faces(frame, known_encodings, known_metadata):
    """Recognize faces in frame with improved accuracy"""
    if frame is None:
        return []
    
    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            return []
        
        # Get face encodings for detected faces
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        matches = []
        for face_encoding in face_encodings:
            # Compare with known faces
            for i, known_encoding in enumerate(known_encodings):
                # Use a stricter tolerance for better accuracy
                match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)[0]
                
                if match:
                    matches.append(known_metadata[i])
                    print(f"MATCH FOUND: {known_metadata[i][1]} (tolerance: 0.6)")
                    break
        
        return matches
        
    except Exception as e:
        print(f"Error in face recognition: {e}")
        return [] 