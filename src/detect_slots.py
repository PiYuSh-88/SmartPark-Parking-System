import cv2
import numpy as np
from src.utils import load_parking_model, preprocess_image, get_grid_regions, extract_slots, format_prediction

def detect_parking_slots(image_path, model, num_rows=5, num_cols=5, threshold=0.5):
    """
    Reads a parking lot image, overlays a grid, predicts occupancy for each slot,
    and returns the annotated image along with the count of empty and occupied slots.
    
    Parameters:
        image_path (str): Path to the input image.
        model (keras.Model): The loaded CNN model.
        num_rows (int): Number of rows in the virtual parking grid.
        num_cols (int): Number of columns in the virtual parking grid.
        threshold (float): Probability threshold for classification.
        
    Returns:
        annotated_image (np.ndarray): The image with bounding boxes drawn.
        empty_count (int): Number of Empty slots.
        occupied_count (int): Number of Occupied slots.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read the image at {image_path}")
        
    annotated_image = image.copy()
    img_height, img_width = image.shape[:2]
    
    # 1. Get grid regions
    regions = get_grid_regions(img_width, img_height, num_rows, num_cols)
    
    # 2. Extract slots
    slots = extract_slots(image, regions)
    
    empty_count = 0
    occupied_count = 0
    
    # 3. Classify and annotate each slot
    for i, slot_img in enumerate(slots):
        x, y, w, h = regions[i]
        
        # Skip empty slices if calculation resulted in 0-size arrays
        if slot_img.size == 0:
            continue
            
        # Preprocess the cropped slot
        processed_slot = preprocess_image(slot_img)
        
        # Predict
        prediction_prob = model.predict(processed_slot, verbose=0)[0][0]
        status = format_prediction(prediction_prob, threshold)
        
        # Determine color (BGR format for OpenCV)
        if status == 'Empty':
            color = (0, 255, 0)  # Green
            empty_count += 1
            label = "E"
        else:
            color = (0, 0, 255)  # Red
            occupied_count += 1
            label = "O"
            
        # Draw bounding box
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, thickness=2)
        
        # Add a small label in the top left corner of the slot
        cv2.putText(annotated_image, label, (x + 5, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return annotated_image, empty_count, occupied_count

if __name__ == '__main__':
    # Simple test code if run directly
    import os
    model_path = 'model/parking_model.h5'
    sample_image = 'dataset/sample/test_lot.jpg'
    
    if os.path.exists(model_path) and os.path.exists(sample_image):
        print("Running sample detection...")
        model = load_parking_model(model_path)
        if model:
            res_img, e_cnt, o_cnt = detect_parking_slots(sample_image, model)
            print(f"Detected {e_cnt} empty and {o_cnt} occupied slots.")
            cv2.imshow("Detection Result", res_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Model or sample image not found. Ensure the dataset and model are present.")
