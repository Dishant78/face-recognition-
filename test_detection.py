#!/usr/bin/env python3
"""
Test script to verify face detection system
"""

from db import init_db, get_all_persons
import cv2
import os

def test_database():
    """Test database functionality"""
    print("=" * 50)
    print("Testing Database")
    print("=" * 50)
    
    # Initialize database
    init_db()
    
    # Get all persons
    persons = get_all_persons()
    
    print(f"Total persons in database: {len(persons)}")
    
    for person in persons:
        print(f"\nPerson ID: {person[0]}")
        print(f"Name: {person[1]}")
        print(f"Age: {person[2]}")
        print(f"Last Seen: {person[3]}")
        print(f"Date Missing: {person[4]}")
        print(f"Additional Info: {person[5]}")
        print(f"Contact Email: {person[6]}")
        print(f"Image Path: {person[7]}")
        
        # Check if image exists
        if os.path.exists(person[7]):
            print(f"✓ Image exists: {person[7]}")
            
            # Try to load image
            image = cv2.imread(person[7])
            if image is not None:
                print(f"✓ Image loaded successfully (Size: {image.shape})")
            else:
                print(f"✗ Failed to load image")
        else:
            print(f"✗ Image not found: {person[7]}")

def test_face_detection():
    """Test face detection on known images"""
    print("\n" + "=" * 50)
    print("Testing Face Detection")
    print("=" * 50)
    
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("✗ Failed to load face cascade classifier")
        return
    
    print("✓ Face cascade classifier loaded successfully")
    
    # Test on each person's image
    persons = get_all_persons()
    
    for person in persons:
        image_path = person[7]
        if os.path.exists(image_path):
            print(f"\nTesting detection on {person[1]}'s image...")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"✗ Failed to load image for {person[1]}")
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                print(f"✓ Found {len(faces)} face(s) in {person[1]}'s image")
                for (x, y, w, h) in faces:
                    print(f"  - Face at ({x}, {y}) with size {w}x{h}")
            else:
                print(f"✗ No faces detected in {person[1]}'s image")

if __name__ == "__main__":
    test_database()
    test_face_detection()
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50) 