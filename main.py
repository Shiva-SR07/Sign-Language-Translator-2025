import cv2
import numpy as np
import tensorflow as tf
import os
import time

def load_model_and_classes():
    """Loads the pre-trained model and class names."""
    # Corrected path to find the model and class names in the 'models' directory
    model = tf.keras.models.load_model("C:\\Sign Language Translator\\models\\asl_model.h5")
    
    class_names = []
    with open("C:\\Sign Language Translator\\models\\class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return model, class_names

def preprocess_image(frame):
    """Preprocesses a single video frame for model prediction."""
    # Resize the image to 64x64 pixels as required by the model
    img = cv2.resize(frame, (64, 64))
    # Normalize pixel values
    img = img / 255.0
    # Add a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def main():
    """Main function to run the real-time sign language translator."""
    try:
        model, class_names = load_model_and_classes()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the 'asl_model.h5' and 'class_names.txt' files are in the 'C:\\Sign Language Translator\\models\\' folder.")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variables for smoothing predictions
    predictions_history = []
    smoothing_window = 10

    print("Application started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a more intuitive mirror-like view
        frame = cv2.flip(frame, 1)

        # Draw a rectangle to define the region of interest (ROI)
        roi = frame[100:350, 100:350]
        cv2.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 2)

        # Preprocess the ROI for prediction
        preprocessed_roi = preprocess_image(roi)
        
        # Get the prediction from the model
        prediction = model.predict(preprocessed_roi, verbose=0)[0]
        
        # Get the class label with the highest probability
        predicted_class_index = np.argmax(prediction)
        
        # Append to history and smooth
        predictions_history.append(predicted_class_index)
        if len(predictions_history) > smoothing_window:
            predictions_history.pop(0)

        # Use the most frequent prediction in the history as the final label
        final_prediction_index = max(set(predictions_history), key=predictions_history.count)
        predicted_label = class_names[final_prediction_index]
        
        # Display the prediction on the frame
        cv2.putText(frame, f'Predicted: {predicted_label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('ASL Sign Language Translator', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()