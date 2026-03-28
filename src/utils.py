import cv2
import numpy as np
import tensorflow as tf

def load_parking_model(model_path):
    """
    Loads the trained Keras model from the provided path.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses the image for the CNN model.
    Resizes, converts color space slightly if needed, and normalizes it.
    """
    # Resize image to match model input shape
    img_resized = cv2.resize(image, target_size)
    
    # Convert from BGR (OpenCV default) to RGB
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_resized
        
    # Normalize pixel values to [0, 1] as configured in ImageDataGenerator
    img_normalized = img_rgb.astype('float32') / 255.0
    
    # Add batch dimension: shape becomes (1, 224, 224, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def get_grid_regions(image_width, image_height, num_rows, num_cols):
    """
    Calculates coordinates for grid-based slot regions across the whole image.
    Returns a list of tuples: (x, y, w, h)
    """
    regions = []
    slot_w = image_width // num_cols
    slot_h = image_height // num_rows
    
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * slot_w
            y = row * slot_h
            regions.append((x, y, slot_w, slot_h))
            
    return regions

def extract_slots(image, regions):
    """
    Extracts cropped image sections from the image based on given regions.
    """
    slots = []
    for (x, y, w, h) in regions:
        slot_img = image[y:y+h, x:x+w]
        slots.append(slot_img)
    return slots

def format_prediction(prediction_probability, threshold=0.5):
    """
    Formats the raw model prediction into a human-readable class label.
    Assuming alphabetical directory naming during training: Empty -> 0, Occupied -> 1.
    """
    if prediction_probability >= threshold:
        return 'Occupied'
    else:
        return 'Empty'
