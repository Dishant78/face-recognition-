import face_recognition
import os
import numpy as np
from PIL import Image

image_path = 'data/dishant.jpg'

if not os.path.exists(image_path):
    print(f'Image not found: {image_path}')
    exit(1)

try:
    # Load and force RGB with Pillow
    pil_img = Image.open(image_path).convert('RGB')
    arr = np.array(pil_img)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    print(f'Loaded image: shape={arr.shape}, dtype={arr.dtype}')
    print(f'Image min/max: {arr.min()}/{arr.max()}')
    print(f'Image type: {type(arr)}, contiguous: {arr.flags["C_CONTIGUOUS"]}')
    encodings = face_recognition.face_encodings(arr)
    if encodings:
        print(f'Success: Found {len(encodings)} face(s) and encoded them.')
    else:
        print('No faces found in the image.')
except Exception as e:
    print(f'Error: {e}')