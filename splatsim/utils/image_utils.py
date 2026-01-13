import cv2
import numpy as np

def letterbox(img, output_size=(224, 224)):
    """
    add black bars to the sides of the image to make it square, then resize
    """
    # Use (c, h, w) input with a numpy array
    c, h, w = img.shape
    scale = min(output_size[1] / w, output_size[0] / h)
    
    # New dimensions that maintain aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 1. Resize the image so the long side fits
    # Opencv expects (h, w, c)
    img_resized = cv2.resize(img.transpose(1, 2, 0), (new_w, new_h), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    
    # 2. Create a black canvas
    # Handles both [0, 1] for float and [0, 255] for uint8
    canvas = np.zeros((3, output_size[0], output_size[1]), dtype=img.dtype)
    
    # 3. Paste the resized image into the center of the canvas
    offset_x = (output_size[1] - new_w) // 2
    offset_y = (output_size[0] - new_h) // 2
    canvas[:, offset_y:offset_y+new_h, offset_x:offset_x+new_w] = img_resized
    
    return canvas