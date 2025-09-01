import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir="data/asl_alphabet_train"):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    images.append(img)
                    labels.append(class_idx)

    images = np.array(images) / 255.0  # Normalize
    labels = np.array(labels)
    return images, labels, class_names

def split_data(images, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test