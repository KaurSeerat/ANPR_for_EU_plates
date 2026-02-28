# AutoScanEU - European Licence Plate Recogntion System

**AutoScanEU** is an end-to-end Automatic Number Plate Recognition (ANPR) system designed for European vehicle plates.
The system integrates deep learning-based object detection (YOLOv8) with Optical Character Recognition (Tesseract OCR) and is deployed via a Flask web application.
This project demonstrates data cleaning, experimental evaluation, performance benchmarking, and analytical problem solving within a computer vision pipeline.

---

## 🚀 Project Overview

This project focuses on:

- Merging and cleaning 1,800+ annotated images to improve model generalisation
- Comparing model performance across multiple datasets
- Optimising detection through hyperparameter tuning
- Designing an OCR preprocessing pipeline
- Conducting structured error analysis
- Deploying a real-time web interface

---

## 📊 Final Performance

After dataset merging and optimisation:

- **mAP@50:** 0.97  
- **Precision:** 0.95  
- **Recall:** 0.94  
- **OCR Mean Character Accuracy:** 68.43%

Dataset merging significantly improved detection accuracy and generalisation.

### Performance Comparison

| Dataset       | mAP@50 | Precision | Recall |
|---------------|--------|-----------|--------|
| Dataset 1     | 0.82   | 0.89      | 0.73   |
| Dataset 2     | 0.87   | 0.89      | 0.76   |
| Merged Data   | 0.97   | 0.95      | 0.94   |

---

## 🔎 Key Analytical Insights

- Dataset diversity directly improved generalisation performance.
- Merging datasets reduced false negatives and improved recall from 0.73 → 0.94.
- OCR preprocessing (thresholding + morphological operations) improved character accuracy by ~15%.
- Most common character confusions:
  - 8 → B  
  - 0 → D  
  - S → 5  
- Performance degradation was strongly linked to lighting variation, motion blur, and perspective distortion.

---

## 🧠 System Architecture

1. User uploads vehicle image via web interface  
2. YOLOv8 detects licence plate  
3. Plate region is cropped and preprocessed  
4. Tesseract OCR extracts alphanumeric text  
5. Recognised plate and bounding box displayed in browser  

---

## 🛠 Tech Stack

### Backend
- Python
- Flask

### Machine Learning & Computer Vision
- YOLOv8 (Ultralytics)
- OpenCV
- Tesseract OCR (pytesseract)

### Data & Analysis
- Pandas
- NumPy
- Matplotlib
- Weights & Biases (experiment tracking)

### Frontend
- HTML
- CSS
- JavaScript

---

## 💻 Running the Application

### 1. Install Dependencies

```bash
pip install flask wandb ultralytics pandas opencv-python numpy matplotlib pytesseract
```
2. To run the code in terminal, use command 'python app.py' or run app.py directly using run in VS code

3. Access app on local browser http://localhost:5000

---

## Project Structure 
```bash 
ANPR_for_EU_plates/
    app/
        app.py               #flask backend
        templates/index.html #frontend UI
        static/
            styles.css       #UI styling
            script.js        #JavaScript for drag & drop, file display
        uploads/             #Uploaded images on website
        results/             #results for YOLO+OCR processed images(uploaded)
    data/
        dataset1             #files from dataset1
        dataset2             #files from dataset2
        merged               #combined dataset1 + dataset2
        
    data_preprocessing/
        load_dataset.py      #Dataset preview and label filtering
        ocr.py               #Main OCR logic with YOLO integration
        yolo_train.py        #Training and Evaluation YOLOv8 model

    models/
        wandb                #showing wandb runs
        yolo-training/       #YOLOv8 training runs and weights
           eu_plate_model14/ #final model training used further for OCR
                    weights/best.pt

    ocr_results/              #Processed OCR images and CSV results
    test_dir/                 #OCR testing images with ground truth
    previewed_dataset/        #YOLO dataset visualisations
    yolo_training.ipynb       #GoogleColab notebook used for training model on YOLOv8s (GPU access)



