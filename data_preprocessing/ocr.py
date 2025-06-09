"""
File: ocr.py
Description: 
This module contains the OCR pipeline for reading licence plates detected by YOLOv8.
It performs preprocessing, applies Tesseract OCR with multiple configurations, 
and returns cleaned and best-matched plates texts.
Also, include tools to evaluate OCR accuracy using ground truth data.
"""
import os
import cv2
import numpy as np
import pytesseract
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO
from difflib import SequenceMatcher
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --------- Configuration ---------
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
model_path = os.path.join(project_root, 'models', 'yolo-training', 'eu_plate_model14', 'weights', 'best.pt')
test_dir = os.path.join(project_root, 'test_dir')
save_folder = os.path.join(project_root, 'ocr_results')
output_csv     = os.path.join(project_root, 'ocr_test_results.csv')
errors_csv     = os.path.join(project_root, 'ocr_common_errors.csv')
histogram_path = os.path.join(project_root, 'ocr_accuracy_histogram.png')
os.makedirs(save_folder, exist_ok=True)


def preprocess_methods(img):
    """
    Applies multiple preprocessing techniques to an image 

    Args:
        img (numpy.ndarray): Original BGR image
    Returns:
        list: List of preprocessed grayscale or binary image
    """
    methods = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    methods.append(gray)

    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    methods.append(filtered)

    _, otsu_bin = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods.append(otsu_bin)

    adapt = cv2.adaptiveThreshold(filtered,
                                255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                19,
                                9)
    methods.append(adapt)

    return methods

def clean_plate_text(text):
    """
    Cleans OCR output by removing unwanted characters and standardising text format.

    Args:
       text (str): Raw OCR output
    Returns:
        str: cleaned and standardised text
    """
    return re.sub(r'[^A-Z0-9 \-ØÆÅÄÖÜ]', '', text.upper()).strip()

def process_image(img_path, output_folder=None, yolo_model=None):
    """
    Process an image to detect and recognize license plates.
    
    Args:
        img_path: Path to input image
        output_folder: Folder to save processed image 
        yolo_model: Pre-loaded YOLO model
    
    Returns:
        tuple: (processed_image_path, detected_texts)
    """
    img = cv2.imread(img_path)
    
    # Load model if not provided
    if yolo_model is None:
        yolo_model = YOLO(model_path)
    
    results = yolo_model.predict(img, conf=0.2)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    plates = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        conf = results[0].boxes.conf[i].item()
        if conf < 0.3:
            continue

        # crop with margin
        mx = int((x2 - x1) * 0.05)
        my = int((y2 - y1) * 0.10)
        x1_, y1_ = max(x1+mx, 0), max(y1+my, 0)
        x2_, y2_ = min(x2-mx, img.shape[1]), min(y2-my, img.shape[0])
        crop = img[y1_:y2_, x1_:x2_]
        if crop.size == 0:
            plates.append('[No Plate Image]')
            continue

        crop = cv2.resize(crop, (300, 100))
        best_text, best_score = '', 0
        configs = [
            "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ØÆÅÄÖÜ",
            "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ØÆÅÄÖÜ",
            "--oem 3 --psm 11"
        ]

        cleaned = ''
        for pre in preprocess_methods(crop):
            for cfg in configs:
                res = pytesseract.image_to_string(pre, config=cfg).strip()
                txt = clean_plate_text(res)
                score = len(txt)
                if 4 <= score <= 10 and score > best_score:
                    best_text, best_score = txt, score
                cleaned = txt or cleaned

        if not best_text and cleaned:
            best_text = cleaned

        plates.append(best_text or '[OCR Failed]')
        # annotate
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, plates[-1], (x1+1, y1-9), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(img, plates[-1], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Determine output path
    if output_folder is not None:
        out_file = os.path.join(output_folder, os.path.basename(img_path))
    else:
        out_file = os.path.join(save_folder, os.path.basename(img_path))
    
    cv2.imwrite(out_file, img)
    
    # Return both path and texts 
    return out_file, plates if plates else ['[No Plates Detected]']

def evaluate_ocr():
    """
    Evaluates OCR performance on a predefined set of test images with known ground truths.
    Outputs:
        - CSV files: accuracy, common errors
        - PNG: histogram of accuracy scores
    """
    model = YOLO(model_path)
    records = []
    error_pairs = []

    # Define correct ground truth for selected images
    gt_dict = {
        "1ca1155083156d72_jpg.rf.be0d654ae14503ba92b55b484e3c96be.jpg": "OUTSTDN",
        "1ead26febde18ce9_jpg.rf.b5935dfe4c2324581a18b32202b9f969.jpg": "ZENDOG",
        "4cb48c8bf41b70a4_jpg.rf.353aad932b92306552d6ad8945154f56.jpg": "TOY DR",
        "4e2cb95b9c509b10_jpg.rf.f5ca3eb3fd46f5d31faa12a94af008af.jpg": "GN64 OTP",
        "747b0f12f54703ee_jpg.rf.eadef55b50b1d1ae46e8b6ed4163c8bc.jpg": "AI7 OVF",
        "car5.jpeg": "59 CADI",
        "dayride_type1_001-mp4-t-85_jpg.rf.26ce6920fb000e3befe633d9072dd087.jpg": "B 99 BNX",
        "dayride_type1_001-mp4-t-1198_jpg.rf.70c7f03a0107b39af396951d803f13dd.jpg": "B 01 XEN",
        "german-car-registration-license-plate-cologne-koln-germany-europe-C6ENJN.jpg": "K SC 124",
        "dayride_type1_001-mp4-t-892_jpg.rf.9ad729b46d2c6a9046a8af30d95643b4.jpg": "B 305 CIL",
        "a929dc75c20da7d8_jpg.rf.38b7b8db79e9b1da92fd90eeaeac8815.jpg": "HAMSTUR",
        "33fa5185-0286-4e8f-b775-46162eba39d4.jpeg": "R8Z D503",
        "eu4.jpeg": "BIMMIAN",
        "images.jpg": "BA24 NED",
        "licensed_car85.jpeg": "TN 52 U 1580",
        "licensed_car148.jpeg": "HP SX 4000",
        "licensed_car171.jpeg": "FLU55H",
        "dayride_type1_002-mp4-t-822_jpg.rf.31fdbebec50d5455041394311048ac3f.jpg": "B 839 HAR",
        "pl_license_plate_343_jpg.rf.bdcebdcca60abbeee8045b58e8ce4f17.jpg": "E3 4230",
        "pl_license_plate_488_jpg.rf.6ad8130c0b8291f97e3e684b215fd598.jpg": "KVT 9263"

    }


    for fn, gt in gt_dict.items():
        img_path = os.path.join(test_dir, fn)
        if not os.path.exists(img_path):
            print(f"Image missing: {fn}")
            continue

        img = cv2.imread(img_path)
        res = model.predict(img, conf=0.3)[0]

        if not res.boxes:
            pred = ''
        else:
            x1, y1, x2, y2 = map(int, res.boxes.xyxy[0])
            crop = cv2.resize(img[y1:y2, x1:x2], (300, 100))
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            raw = pytesseract.image_to_string(
                bw,
                config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ØÆÅÄÖÜ'
            )
            pred = clean_plate_text(raw)

        acc = round(SequenceMatcher(None, pred, gt).ratio() * 100, 2)
        records.append({
            'Image': fn,
            'Ground Truth': gt,
            'OCR Output': pred,
            'Character Accuracy (%)': acc
        })

        for p, t in zip(pred, gt):
            if p != t:
                error_pairs.append(f"{p}->{t}")

    df = pd.DataFrame(records)
    df.to_csv("ocr_20_image_results.csv", index=False)

    cnt = Counter(error_pairs)
    pd.DataFrame(cnt.items(), columns=["Pair", "Count"]).to_csv("ocr_20_image_errors.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(df["Character Accuracy (%)"], bins=10, edgecolor='black')
    plt.title(f"OCR Accuracy Distribution ({len(df)} Images)")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig("ocr_20_image_histogram.png")
    print("Evaluation complete. Files saved:")
    print(" - ocr_20_image_results.csv")
    print(" - ocr_20_image_errors.csv")
    print(" - ocr_20_image_histogram.png")

if __name__ == '__main__':
    evaluate_ocr()