# AutoScanEU - European Licence Plate Recogntion System

**AutoScanEU** is an AI-powered web application for automatic licence plate recognition (ANPR) across European countries. It uses a YOLOv8-based object detector and Tesseract OCR to identify and extract licence plate numbers from vehicle images.

## Features
- Upload vehicle images via a simple web interface.
- Detects number plates in real-time using a YOLOv8 model.
- Performs OCR to extract plate numbers from detected regions.
- Displays original, processed images and recognised plates.
- Supports evaluation of OCR accuracy and common errors.
- Interactive drag & drop UI for easy use.

## Tech Stack
**Frontend**: HTML, CSS, JavaScript
**Backened**: Flask (Python)
**Model**: YOLOv8 (from Ultralytics) for plate detection
**OCR**: Tesseract OCR via `pytesseract`
**Visualisation**: `matplotlib`, `OpenCV`

## To run the application
1. **Install Dependencies**  
   You can install all dependencies using pip:
  

   ```bash
    pip install flask wandb ultralytics pandas opencv-python numpy matplotlib pytesseract

2. To run the code in terminal, use command 'python app.py' or run app.py directly using run in VS code

3. Access app http://localhost:5000 

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



