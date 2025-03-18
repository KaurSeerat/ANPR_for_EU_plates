# import pytesseract
# import torch
# import numpy as np
# from ultralytics import YOLO
# import os, cv2, re

# # Load YOLO model
# yolo_model = YOLO("./yolo-training/eu_plate_model4/weights/best.pt")

# # Function to preprocess plate image for OCR
# def process_plate_for_ocr(plate_img):
#     """Preprocesses the detected plate image to enhance OCR accuracy."""
#     if plate_img.shape[0] < 10 or plate_img.shape[1] < 10:
#         return None

#     plate_img = cv2.resize(plate_img, (300, 100))
#     gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
#     thresh_plate_img = cv2.adaptiveThreshold(gray_plate_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                              cv2.THRESH_BINARY, 11, 2)
#     kernel = np.ones((3, 3), np.uint8)
#     morph_plate_img = cv2.morphologyEx(thresh_plate_img, cv2.MORPH_OPEN, kernel, iterations=1)
#     smooth_plate_img = cv2.medianBlur(morph_plate_img, 3)

#     return smooth_plate_img

# # Load image for testing
# img_path = "image2.jpg"
# img = cv2.imread(img_path)

# # Run YOLO detection
# results = yolo_model.predict(img, conf=0.3)

# # Extract bounding boxes
# boxes_tensor = results[0].boxes.xyxy
# boxes = boxes_tensor.cpu().numpy()

# detected_plates = []

# for i, box in enumerate(boxes):
#     x1, y1, x2, y2 = map(int, box[:4])
#     confidence = results[0].boxes.conf[i].item()

#     if confidence < 0.3:
#         print(f"Skipping detection with low confidence: {confidence:.2f}")
#         continue

#     # Crop detected plate
#     plate_img = img[y1:y2, x1:x2]
#     processed_plate = process_plate_for_ocr(plate_img)

#     if processed_plate is None:
#         print("Cropped plate too small for OCR, skipping.")
#         continue

#     # Run OCR
#     plate_text = pytesseract.image_to_string(processed_plate, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
#     plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)

#     if plate_text:
#         detected_plates.append(plate_text)
#         print(f"Detected Plate: {plate_text}")

#     # Draw bounding box and label
#     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

# # Save result image
# output_path = './output7.jpg'
# cv2.imwrite(output_path, img)

# # Print final detected plates
# print("Detected License Plates:", detected_plates)

import cv2
import numpy as np
import pytesseract
import os
import re
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("./yolo-training/eu_plate_model9/weights/best.pt")

def process_plate_for_ocr(plate_img):
    """Preprocesses the detected plate image to enhance OCR accuracy."""
    if plate_img.shape[0] < 10 or plate_img.shape[1] < 10:
        return None

    plate_img = cv2.resize(plate_img, (300, 100))
    gray_plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    thresh_plate_img = cv2.adaptiveThreshold(
        gray_plate_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    morph_plate_img = cv2.morphologyEx(thresh_plate_img, cv2.MORPH_OPEN, kernel, iterations=1)
    smooth_plate_img = cv2.medianBlur(morph_plate_img, 3)

    return smooth_plate_img

def process_image(img_path, save_folder):
    """Processes the image with YOLO and OCR."""
    img = cv2.imread(img_path)
    results = yolo_model.predict(img, conf=0.2)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    detected_plates = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        confidence = results[0].boxes.conf[i].item()

        if confidence < 0.3:
            continue

        plate_img = img[y1:y2, x1:x2]
        processed_plate = process_plate_for_ocr(plate_img)

        if processed_plate is None:
            continue

        plate_text = pytesseract.image_to_string(
            processed_plate, config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)

        if plate_text:
            detected_plates.append(plate_text)
            print(f"Detected Plate: {plate_text}")

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    output_filename = os.path.join(save_folder, os.path.basename(img_path))
    cv2.imwrite(output_filename, img)

    return output_filename, detected_plates
