from PIL import Image
import sys

# Change this to your image filename
input_path = 'data/1myface.jpg'
output_path = 'data/1myface_clean.jpg'

img = Image.open(input_path).convert('RGB')
img.save(output_path, 'JPEG')
print(f'Converted {input_path} to {output_path} as standard JPEG.') 