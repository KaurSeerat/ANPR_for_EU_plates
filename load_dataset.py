"""
File: load_preview_dataset.py
Description:
This script loads YOLO images and labels from a dataset and allows manual visualization of 
a few samples with bounding boxes drawn on the images. Useful for verifying data preprocessing.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to load images and labels manually
def load_yolo_images_and_labels(images_folder, labels_folder, img_size=(640, 640)):
    images = []
    labels = []
    for image_file in os.listdir(images_folder):
        img_path = os.path.join(images_folder, image_file)
        
        # Extract base name without extension for label matching
        base_name = image_file.split('.')[0]
        label_file_name = f"{base_name}_jpg.rf.*.txt"  # Match the label file name with the suffix
        
        # Search for label files that match the base name
        label_path = None
        for label_file in os.listdir(labels_folder):
            if label_file.startswith(base_name) and label_file.endswith('.txt'):
                label_path = os.path.join(labels_folder, label_file)
                break
        
        # Proceed if label file is found
        if label_path:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size) / 255.0  # Normalize
                images.append(img)
                
                # Parse YOLO labels
                with open(label_path, 'r') as file:
                    label_data = file.readlines()
                    bboxes = []
                    for label in label_data:
                        parts = label.strip().split()
                        if len(parts) == 5:  
                            class_id = int(parts[0])
                            bbox = list(map(float, parts[1:]))  # Normalized bbox coordinates
                            bboxes.append([class_id] + bbox)  # Ensure class_id is included
                        else:
                            print(f"Skipping invalid label: {label.strip()}")
                    labels.append(bboxes)
            else:
                print(f"Could not read image {img_path}")
        else:
            print(f"Label for {image_file} not found")
    
    return np.array(images), labels

# Load a subset of data (e.g., 5 images from the training set)
def preview_dataset(images, labels, num_samples=5):
    for i in range(min(num_samples, len(images))):
        img = images[i]
        label = labels[i]
        
        # Rescale image back to original size for visualization
        img = (img * 255).astype(np.uint8)
        
        # Plot the bounding boxes
        for box in label:
            if len(box) == 5:  # Ensure there are 5 values to unpack
                class_id, x_center, y_center, width, height = box
                img_height, img_width, _ = img.shape
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                x_max = int((x_center + width / 2) * img_width)
                y_max = int((y_center + height / 2) * img_height)
                
                # Draw the bounding box
                img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                # Optionally add class_id text
                cv2.putText(img, str(class_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                print(f"Skipping bounding box with invalid format: {box}")
        
        # Display the image with bounding boxes
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        plt.axis("off")
        plt.show()

if __name__ == "__main__":       
# Example usage
    dataset_path = "./dataset/"

    X_train, y_train = load_yolo_images_and_labels(
        os.path.join(dataset_path, "train", "images"),
        os.path.join(dataset_path, "train", "labels")
    )

    # Preview some training samples
    preview_dataset(X_train, y_train,num_samples=6)
    print('dataset previewed')
