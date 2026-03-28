import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Dense, Flatten

def build_model(input_shape=(224, 224, 3)):
    """
    Builds a CNN model for binary classification (Empty vs Occupied).
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def plot_metrics(history, save_path='model/training_metrics.png'):
    """
    Plots and saves the training accuracy and loss curves.
    """
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    if val_acc:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if val_loss:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")

def main():
    train_dir = 'dataset/train'
    valid_dir = 'dataset/valid'
    test_dir = 'dataset/test'
    
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 10
    
    print("Setting up data generators...")
    # Training Data Generator with Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True
    )
    
    # Validation Data Generator (only rescaling)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    # Check if directories exist and have contents before starting flow
    if not os.path.exists(train_dir) or not any(os.scandir(train_dir)):
        print(f"Warning: Training directory '{train_dir}' is empty or does not exist. Please add data to train.")
        return
        
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    
    if os.path.exists(valid_dir) and any(os.scandir(valid_dir)):
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
    else:
        valid_generator = None
        print(f"Warning: Validation directory '{valid_dir}' is empty. Validation step will be skipped during training.")

    if os.path.exists(test_dir) and any(os.scandir(test_dir)):
        test_generator = valid_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )
    else:
        test_generator = None

    print("Building model...")
    model = build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model.summary()
    
    print("Starting training...")
    try:
        # Check if there are actual files matched by the generator
        if train_generator.samples == 0:
            print("No images found in training directory. Aborting training.")
            return

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=valid_generator,
            verbose=1
        )
        
        # Save Model
        os.makedirs('model', exist_ok=True)
        model_path = 'model/parking_model.h5'
        model.save(model_path)
        print(f"Model successfully saved to {model_path}")
        
        # Show Metrics
        plot_metrics(history)
        
    except Exception as e:
        print(f"Training failed or encountered an error: {e}")

if __name__ == '__main__':
    main()
