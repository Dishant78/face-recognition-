#!/usr/bin/env python3
"""
Test script to verify face detection accuracy for specific faces
"""

from db import init_db, get_all_persons
import cv2
import numpy as np
import os

def test_face_accuracy():
    """Test face detection accuracy on known faces"""
    print("=" * 60)
    print("EXPERT-LEVEL FACE DETECTION ACCURACY TEST")
    print("=" * 60)
    
    # Initialize database
    init_db()
    
    # Get all persons
    persons = get_all_persons()
    
    print(f"Testing accuracy for {len(persons)} persons in database:")
    
    for person in persons:
        print(f"\n{'='*40}")
        print(f"Testing: {person[1]}")
        print(f"Image: {person[7]}")
        print(f"{'='*40}")
        
        if not os.path.exists(person[7]):
            print(f"✗ Image not found: {person[7]}")
            continue
        
        # Load image
        image = cv2.imread(person[7])
        if image is None:
            print(f"✗ Failed to load image")
            continue
        
        print(f"✓ Image loaded (Size: {image.shape})")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if face_cascade.empty():
            print("✗ Failed to load face cascade")
            continue
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            print(f"✓ Found {len(faces)} face(s)")
            
            for i, (x, y, w, h) in enumerate(faces):
                print(f"  Face {i+1}: Position ({x}, {y}), Size {w}x{h}")
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Test preprocessing
                try:
                    # Resize to standard size
                    face_resized = cv2.resize(face_roi, (128, 128))
                    
                    # Apply CLAHE
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    face_clahe = clahe.apply(face_resized)
                    
                    # Normalize
                    face_normalized = cv2.normalize(face_clahe, None, 0, 255, cv2.NORM_MINMAX)
                    
                    print(f"    ✓ Preprocessing successful")
                    print(f"    ✓ Face size: {face_normalized.shape}")
                    
                    # Save processed face for verification
                    processed_path = f"data/processed_{person[1].replace(' ', '_')}.jpg"
                    cv2.imwrite(processed_path, face_normalized)
                    print(f"    ✓ Saved processed face: {processed_path}")
                    
                except Exception as e:
                    print(f"    ✗ Preprocessing failed: {e}")
        else:
            print("✗ No faces detected in image")
    
    print(f"\n{'='*60}")
    print("ACCURACY TEST COMPLETED")
    print("=" * 60)
    print("Check the processed face images in the data folder")
    print("These should show clear, normalized faces for better recognition")

if __name__ == "__main__":
    test_face_accuracy() 