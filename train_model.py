import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train():
    """Trains a CNN model for ASL sign language recognition."""

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'asl_alphabet_train')
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'asl_model.h5')
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Dataset directory not found at: {data_dir}")

    # Use ImageDataGenerator to load data in batches to save memory
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2, # Use 20% of data for validation
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Get class names from the generator
    class_names = list(train_generator.class_indices.keys())

    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting model training. This will take a while...")
    
    # Train the model using the generators
    model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )

    # Save the trained model and class names
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # You also need to save class names for the main application
    class_names_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for item in class_names:
            f.write("%s\n" % item)

    return class_names

if __name__ == '__main__':
    class_names = train()
    print("Training completed successfully.")