from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def create_diamond_mask(size, diamond_size=0.5, rotation=0):
    h, w = size
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    
    # Create diamond shape
    mask = np.abs(x - center_x) / (w * diamond_size) + np.abs(y - center_y) / (h * diamond_size) < 1
    
    # Rotate the mask
    mask = rotate(mask, rotation, reshape=False, order=1)
    
    return mask.astype(float)

def reflect_image(img):
    return np.flip(img, axis=(0, 1))

def diamond_reflection_effect(image, diamond_size=0.5, edge_softness=20, rotation=0):
    img_array = np.array(image).astype(float) / 255

    # Create the diamond mask with rotation
    mask = create_diamond_mask(img_array.shape[:2], diamond_size, rotation)
    
    # Create reflected image for corners
    reflected = reflect_image(img_array)
    
    # Apply strong Gaussian blur to the mask for soft edges
    soft_mask = gaussian_filter(mask, sigma=edge_softness)
    
    # Apply the soft mask
    result = img_array * np.expand_dims(soft_mask, -1) + reflected * np.expand_dims(1 - soft_mask, -1)
    
    # Add dreamy effect
    result = result * 0.8 + 0.2  # Increase brightness
    result = np.clip(result, 0, 1)
    
    # Add slight color fringe for chromatic aberration
    fringe = np.roll(result, 3, axis=1)
    result = result * 0.9 + fringe * 0.1
    
    # Convert back to uint8
    result = (result * 255).astype(np.uint8)
    return Image.fromarray(result)

@app.route('/')
def home():
    return "Diamond Reflection Effect API"

@app.route('/diamond_reflection_effect', methods=['POST'])
def diamond_reflection():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file.stream)
    
    diamond_size = float(request.form.get('diamond_size', 0.5))
    edge_softness = int(request.form.get('edge_softness', 20))
    rotation = float(request.form.get('rotation', 0))
    
    output_image = diamond_reflection_effect(image, diamond_size, edge_softness, rotation)
    
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)