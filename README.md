# SmartPark - Smart Parking System using Machine Learning and Computer Vision

SmartPark is a complete, end-to-end machine learning project designed to detect whether parking slots are Empty or Occupied from images. It integrates a Convolutional Neural Network (CNN) for image classification, OpenCV for spatial processing (grid extraction and annotation), and PyQt5 for a desktop GUI application.

This project is structured for readability and educational purposes, making it an excellent resource for learning real ML workflows, from data augmentation and CNN training to inference and application integration.

## Project Structure

```text
SmartPark/
│
├── dataset/                    # Data directory
│   ├── train/                  # Training data
│   │   ├── Empty/              # Empty slot images
│   │   └── Occupied/           # Occupied slot images
│   ├── valid/                  # Validation data
│   ├── test/                   # Testing and evaluation data
│   └── sample/                 # Small sample images for trying the GUI
│
├── model/                      # Saved models and training metrics
│   └── parking_model.h5        # Trained Keras model (generated via train_model.py)
│
├── src/                        # Core source code
│   ├── prepare_dataset.py      # Parses labels.json and categorizes raw image dataset
│   ├── train_model.py          # Script to build and train the CNN model
│   ├── detect_slots.py         # OpenCV image processing and inference module
│   ├── gui_app.py              # PyQt5 graphical desktop application
│   └── utils.py                # Helper functions for processing and formatting
│
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignorations rule
```

## System Architecture

1. **Dataset Pipeline:** Images are loaded and dynamically augmented (rotation, zooming, horizontal flipping) using TensorFlow's `ImageDataGenerator`.
2. **CNN Model:** A custom Deep Learning architecture processes 224x224 RGB images using blocks of `Conv2D`, `BatchNormalization`, and `MaxPooling2D` layers, culminating in `Dense` and `Dropout` layers to perform binary classification (Empty vs. Occupied).
3. **Computer Vision Integration:** OpenCV artificially overlays a virtual geometric grid onto a full parking lot image, cropping it into individual "slots." Each slot is classified via the trained CNN model.
4. **GUI Desktop Application:** A streamlined PyQt5 app that allows users to seamlessly upload a parking image, analyze it using the trained system, and visibly display the annotated outcome along with a quantitative summary.

## Technologies Used

* **Python 3.x**
* **TensorFlow / Keras :** Model building, dataset pipelining, training.
* **OpenCV (`opencv-python`) :** Image manipulation, bounded rectangle drawing, grid sectioning.
* **PyQt5 :** Cross-platform desktop UI framework.
* **Matplotlib :** Metric visualization (Accuracy / Loss charts).
* **NumPy :** Numerical operations and array abstractions.

## How to Setup and Run

### 1. Installation

Clone this repository and open the folder in an IDE like VS Code.
Ensure you have a modern Python environment installed.

```bash
# Optional: Create a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
Add your training, validation, and test datasets under the `dataset/` directory. Each split should possess an `images/` directory alongside a unifying `labels.json` file determining classification.

```text
dataset/
├── train/
│   ├── images/
│   └── labels.json
...
```

**Step 1:** Prepare the dataset by auto-categorizing images based on the provided JSON parameters:
```bash
python src/prepare_dataset.py
```

### 3. Model Training
Once the dataset is ready (`dataset/train/Empty` and `dataset/train/Occupied` mapped successfully via the preparation script), train the model by running:

**Step 2:**
```bash
python src/train_model.py
```

*This will read data using `ImageDataGenerator`, train a CNN classifier, save metric plots to `model/training_metrics.png`, and automatically save the best model weights as `model/parking_model.h5`.*

### 4. Running the Parking Detection Application
To launch the GUI and test the trained model on full parking lot images:

```bash
python src/gui_app.py
```

*Upload a parking lot image directly through the interface, click **Detect Parking**, and watch the magic happen! Empty slots will be highlighted in **GREEN** and Occupied slots in **RED**.*

## Example Workflow

- **Load Data & Train:** We supply individual cropped slot images of cars vs empty spaces to help our CNN identify what constitutes "occupied."
- **Inference with Vision:** We feed the system a *macro* image of a large parking array. The cv2 script divides this image into a uniform grid (e.g., 5x5).
- **Yield Results:** Each component in the grid is passed down to the model. We collate the results and render a complete overview on the graphical interface. 
