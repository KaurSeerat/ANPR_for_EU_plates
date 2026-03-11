# AutoScanEU  
### Deep Learning Based European Licence Plate Recognition System

AutoScanEU is an end to end Automatic Number Plate Recognition (ANPR) system designed to detect and recognise European vehicle licence plates under real world conditions.

The system integrates deep learning based object detection using YOLOv8 with an Optical Character Recognition pipeline powered by OpenCV and Tesseract. The project investigates how dataset diversity, preprocessing techniques and model optimisation affect detection accuracy and recognition performance.

The system is deployed through a Flask based web application that allows users to upload images and receive real time licence plate detection and recognition results.

---

# Research Motivation

Automatic Number Plate Recognition systems are widely used in:

- Smart parking systems  
- Traffic monitoring  
- Law enforcement  
- Automated toll collection  

European licence plates present unique challenges due to:

- Multiple formats across countries  
- Variation in fonts and character sets  
- Inconsistent lighting conditions  
- Skewed or partially occluded plates  

This project investigates how modern computer vision techniques can improve licence plate detection and recognition performance across diverse European plate formats.

---

# System Architecture

The system follows a multi stage computer vision pipeline.

1. User uploads a vehicle image through a web interface  
2. YOLOv8 detects the licence plate region  
3. The detected region is cropped and preprocessed  
4. OCR extracts the alphanumeric characters  
5. The recognised plate and bounding box are displayed to the user  

Pipeline:

```
Image → YOLOv8 Detection → OpenCV Preprocessing → Tesseract OCR → Web Output
```

---

# Methodology

## Dataset Preparation

Two annotated datasets were sourced and merged to improve model generalisation.

Key preprocessing steps included:

- Merging datasets to increase diversity  
- Removing inconsistent class labels  
- Dataset visualisation for label verification  
- Train validation test split  
- Data augmentation including rotation, brightness variation and noise injection  

Final dataset size:

**≈ 1,800 annotated licence plate images**

---

## Model Training

The detection model was trained using the Ultralytics YOLOv8 framework.

Training configuration included:

- YOLOv8n and YOLOv8s architecture comparison  
- Transfer learning using pretrained weights  
- Hyperparameter tuning  
- Monitoring training and validation loss  

Performance evaluation metrics included:

- Precision  
- Recall  
- Mean Average Precision (mAP)  
- Training and validation loss curves  

Experiment tracking was conducted using **Weights and Biases**.

---

## OCR Pipeline

After licence plate detection, a preprocessing pipeline was applied before OCR.

Steps included:

- Grayscale conversion  
- Adaptive thresholding  
- Morphological operations  
- Noise reduction  

Tesseract OCR was configured with a restricted character set consisting of uppercase letters and digits to reduce recognition errors.

---

# Experimental Results

After dataset merging and model optimisation:

| Metric | Result |
|------|------|
| mAP@50 | 0.97 |
| Precision | 0.95 |
| Recall | 0.94 |
| OCR Mean Character Accuracy | 68.43% |

Dataset diversity significantly improved model performance and reduced false negative detections.

---

# Performance Comparison

| Dataset | mAP@50 | Precision | Recall |
|------|------|------|------|
| Dataset 1 | 0.82 | 0.89 | 0.73 |
| Dataset 2 | 0.87 | 0.89 | 0.76 |
| Merged Dataset | 0.97 | 0.95 | 0.94 |

The merged dataset improved recall from **0.73 to 0.94** by increasing training diversity and reducing model bias towards specific plate formats.

---

# Error Analysis

Analysis of OCR output revealed several recurring recognition errors.

Common character confusions:

- `8` interpreted as `B`  
- `0` interpreted as `D`  
- `S` interpreted as `5`  

Performance degradation was most commonly caused by:

- Low lighting conditions  
- Motion blur  
- Perspective distortion  
- Reflective plate surfaces  

These findings suggest that further improvements could be achieved through a deep learning based OCR model instead of rule based OCR.

---

# Technology Stack

### Programming
- Python

### Machine Learning
- YOLOv8  
- PyTorch  

### Computer Vision
- OpenCV  

### OCR
- Tesseract OCR  

### Experiment Tracking
- Weights and Biases  

### Backend
- Flask  

### Frontend
- HTML  
- CSS  
- JavaScript  

### Data Analysis
- Pandas  
- NumPy  
- Matplotlib  

---

# Running the Application

### 1. Install Dependencies

```bash
pip install flask wandb ultralytics pandas opencv-python numpy matplotlib pytesseract
```

### 2. Run the Application

```bash
python app.py
```

### 3. Access in Browser

```
http://localhost:5000
```

Upload an image containing a vehicle licence plate to test the detection and recognition pipeline.

---

# Project Structure

```
ANPR_for_EU_plates/
    app/
        app.py
        templates/
            index.html
        static/
            styles.css
            script.js
        uploads/
        results/

    data/
        dataset1/
        dataset2/
        merged/

    data_preprocessing/
        load_dataset.py
        ocr.py
        yolo_train.py

    models/
        yolo-training/
        wandb/

    ocr_results/
    test_dir/
    previewed_dataset/

    yolo_training.ipynb
```

---

# Future Work

Several improvements could further enhance system performance.

- Training a CNN based OCR model instead of Tesseract  
- Expanding dataset diversity across more European countries  
- Implementing real time video based licence plate detection  
- Improving robustness under extreme lighting conditions  
- Integrating multilingual plate recognition  

---

# Academic Work

This project was developed as part of a **BSc Computing Science dissertation at the University of East Anglia**.

The full dissertation is not publicly available due to university policies. It can be provided upon request.

---

# Author

**Seerat Kaur**

Computing Science Graduate  
Machine Learning | Computer Vision | Data Science  

LinkedIn: https://www.linkedin.com/in/seerat-kaur-4878bb249/  
GitHub: https://github.com/KaurSeerat
