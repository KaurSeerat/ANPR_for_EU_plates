# AutoScanEU — ANPR for European Licence Plates

End-to-end Automatic Number Plate Recognition system for European vehicles, combining YOLOv8 object detection with a Tesseract OCR pipeline, deployed as a Flask web application. Developed as a First-Class BSc Computing Science dissertation at the University of East Anglia (2025).

> **Detection results:** mAP@50 of **0.97** · Precision **0.95** · Recall **0.94** · Dataset merging reduced false negatives by **21%** vs. single-dataset training

---

## 🌐 Web Application — AutoScanEU

Users upload a vehicle image through the browser. The system automatically detects licence plates, crops the region, runs the OCR pipeline, and displays the processed image with bounding boxes and extracted plate text, no Python knowledge required.

<img width="900" height="900" alt="image" src="https://github.com/user-attachments/assets/eb15ed26-a054-49bc-86ce-eda716d259f7" />

---
**Example output — incorrect vs correct OCR recognition:**

| Challenging conditions (partial error) | Ideal conditions (correct) |
|---|---|
| <img width="310" height="162" alt="Incorrect OCR - GN64OTP misread as GN4OTF" src="https://github.com/user-attachments/assets/e29e5514-434c-42ec-bbde-cb4402bbfb49" /> | <img width="310" height="162" alt="Correct OCR - BA24 NED" src="https://github.com/user-attachments/assets/0929d79f-5ae8-44d9-8be6-30aaa42140f4" /> |

> Left: "GN64OTP" partially misread as "GN4OTF" due to perspective distortion and lighting glare, a known Tesseract limitation with angled plates. Right: "BA24 NED" recognised correctly under ideal lighting.

---
## 📐 System Architecture

```
Image Upload → YOLOv8 Detection → Bounding Box Crop → OpenCV Preprocessing → Tesseract OCR → Result Display
```

The system has three loosely coupled modules, detection, OCR, and web interface allowing any component to be swapped independently (e.g. replacing Tesseract with a CNN-based OCR without touching the Flask layer).

---

## 📊 Model Performance

### Dataset Comparison — Before and After Merging

| Metric | Dataset 1 (385 imgs) | Dataset 2 (1,455 imgs) | Merged Dataset |
|---|---|---|---|
| Model | YOLOv8n | YOLOv8s | YOLOv8s |
| Epochs | 20 | 50 | 50 |
| mAP@50 | 0.82 | 0.87 | **0.97** |
| mAP@50-95 | 0.609 | 0.710 | **0.896** |
| Precision | 0.89 | 0.89 | **0.95** |
| Recall | 0.73 | 0.76 | **0.94** |
| Validation Box Loss | ~0.94 | ~0.52 | **~0.45** |
| Overfitting | Yes | No | No |
| Generalisation | Poor | Good | **Best** |

Dataset 1 alone showed signs of overfitting (validation loss stagnant at ~0.94 while training loss dropped). Merging both datasets improved recall from 0.73 to 0.94, reducing false negatives by 21% by exposing the model to greater plate diversity across European countries.

---
### Model Performance Metrics (Merged Dataset)
<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/5b140aa8-aa20-4710-9d09-adeee65d2d0c" />


### Training Convergence

<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/213d486f-2d2b-4487-b0cb-ee9c332ef470" />

> Training vs. validation loss curves across 50 epochs. The narrow gap between curves confirms strong convergence and no overfitting. Both box loss values converge towards 0.5–0.6.

### Precision-Recall and Confidence Curves

<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/e65f1dd7-fb5a-4284-9fda-bbbc08492c1b" />

<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/b320688a-076e-4063-b6b3-ac51adf9fe93" />

<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/a402b646-1361-4e99-9c30-f210b4d4c342" />

> Precision approaches 1.0 at high confidence thresholds. Recall remains high across most thresholds. The precision-recall curve area confirms strong detection capability with minimal false positives.

### Confusion Matrix

<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/9b5b554b-856e-4d6c-ad9a-130b4936ed8f" />
(confusion_matrix)

> 395 correct licence plate detections (true positives). Some misclassification between background and plates in edge cases — expected given image diversity.

---

## 🔍 Detection Examples

### Bounding Box Detection — Training Data
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/c8bc348b-6e26-4576-830f-ed5b7d9890a3" />

Train data detection

### Bounding Box Detection — Validation Data
<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/38df3a5e-6fe5-4e8b-8fc2-8acc9ee2c8f0" />

Validation data detection

### Before and After Class Filtering

| Before filtering (class 0 + 1) | After filtering (class 0 only) |
|---|---|
| <img width="400" height="400" alt="Before filtering - class 0 and 1" src="https://github.com/user-attachments/assets/d16c4ce1-50db-466a-a885-4fb9dd0ec787" /> | <img width="400" height="400" alt="After filtering - class 0 only" src="https://github.com/user-attachments/assets/cb5d6494-a8a4-46f1-a3d2-923f48f9e3d8" /> |

---

## ⚙️ OCR Pipeline

After YOLOv8 detects a plate, each bounding box region passes through a 5-stage preprocessing pipeline before text extraction:

```
Crop plate region → Resize (300×100px) → Grayscale → Bilateral filter + Otsu/Adaptive thresholding → Morphological opening → Tesseract OCR → Regex post-processing
```

### Pipeline Visualised

**1. Plate crop**

<img width="676" height="348" alt="image" src="https://github.com/user-attachments/assets/aad61361-badf-4d66-9f39-93a01597a1fc" />
(ocr_results/plate_0_cropped.jpg)

**2. Grayscale conversion**

<img width="300" height="100" alt="image" src="https://github.com/user-attachments/assets/c3c034f3-a8c3-4b81-9ad7-64d2b1c915b8" />
(ocr_results/plate_0_gray.jpg)

**3. Otsu's thresholding (binarisation)**

<img width="300" height="100" alt="image" src="https://github.com/user-attachments/assets/bec19805-5bb2-4c61-a26b-34df06decc8f" />
(ocr_results/plate_0_otsu.jpg)

### Tesseract Configuration
- `-psm 7` — assumes single line of text (optimal for number plates)
- `tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-ØÆÅÄÖÜ`
- Output ranked by length within expected range of 4–10 characters
- Regex post-processing: removes non-alphanumeric characters

### OCR Results Summary

| Metric | Value |
|---|---|
| Total images evaluated | 20 |
| Mean character accuracy | **68.43%** |
| Median character accuracy | 74.84% |
| Images with ≥ 80% accuracy | 9 / 20 |
| Images with < 50% accuracy | 3 / 20 |

Preprocessing pipeline (adaptive thresholding + morphological operations) improved accuracy by **15%** compared to raw images. Performance plateaued for motion-blurred and angled plates.

### OCR Accuracy Distribution

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/d6edddf1-7d27-46fb-8214-555bbc4c2689" />

> Concentration of images around 60–90% accuracy range. Low-performing samples (< 50%) associated with low contrast, glare, or unusual plate designs.

### Most Common Character Errors

| Confusion | Cause |
|---|---|
| `8` → `B` | Visual shape similarity |
| `0` → `D` | Visual shape similarity |
| `S` → `5` | Visual shape similarity |

These three confusions account for **62% of all OCR errors** and stem from Tesseract's reliance on shape matching rather than context.

---

## 🧪 System Testing

| Test Case | Result | Notes |
|---|---|---|
| Clear plate image | ✅ Pass | Correct extraction under ideal conditions |
| Angled plate | ✅ Pass | Minor OCR misread (e.g. 'O' vs '0') — expected |
| Night-time image | ❌ Fail | Low-light reduces accuracy; histogram equalisation would help |
| Blurry image | ❌ Fail | Motion blur defeats Tesseract; needs deblurring step |
| Occluded plate | ✅ Pass | No detection — handles occlusions gracefully |
| Non-plate image | ✅ Pass | No false positives |
| Multiple vehicles | ✅ Pass | All plates detected; OCR struggles with small plates |
| Special chars (Ä, Ö) | ⚠️ Partial | Tesseract whitelist needs expansion for all EU characters |

---

## 🛠️ Technology Stack

| Layer | Tools |
|---|---|
| Detection | YOLOv8s (Ultralytics), PyTorch, transfer learning |
| OCR | Tesseract (pytesseract), OpenCV |
| Experiment tracking | Weights & Biases (W&B) |
| Web app | Flask, HTML/CSS, Bootstrap |
| Data analysis | Pandas, NumPy, Matplotlib |
| Environment | Google Colab (GPU training), VS Code, Anaconda |

**Training environment:** Final training migrated from local CPU (limited by memory overflow on YOLOv8s) to Google Colab GPU — enabling batch size of 8 and 50-epoch runs without memory constraints.

### Weights & Biases — Resource Monitoring

<img width="1000" height="900" alt="image" src="https://github.com/user-attachments/assets/f876f3ba-2c27-4a57-9afb-ab9d4aa12aa3" />

> GPU, CPU, memory, and network traffic tracked across training runs during initial experiments.

---

## 📁 Repository Structure

```
ANPR_for_EU_plates/
├── app/
│   ├── app.py                  # Flask application entry point
│   ├── templates/index.html    # Web UI
│   └── static/                 # CSS, JS, screenshot
├── data/
│   ├── dataset1/               # Roboflow EU plates (385 images, class 0+1)
│   ├── dataset2/               # Roboflow EU plates (1,455 images, class 0)
│   └── merged/                 # Final unified training dataset (~1,800 images)
├── data_preprocessing/
│   ├── load_dataset.py         # Image loading, normalisation, label parsing
│   ├── ocr.py                  # OCR pipeline (crop, preprocess, Tesseract)
│   └── yolo_train.py           # YOLOv8 training script
├── models/yolo-training/       # Saved weights, loss curves, W&B logs
├── ocr_results/                # OCR accuracy CSV, character error analysis
├── previewed_dataset/          # Bounding box visualisations
├── test_dir/                   # Test images
├── yolo_training.ipynb         # Training notebook (Colab)
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Application

```bash
git clone https://github.com/KaurSeerat/ANPR_for_EU_plates
cd ANPR_for_EU_plates
pip install -r requirements.txt
python app/app.py
# Open http://localhost:5000 in your browser
# Upload a vehicle image → click Detect Plates → view result
```

---

## ⚠️ Limitations

- **YOLO bounding box alignment:** Occasionally too tight or misaligned, affecting crop quality for OCR
- **Tesseract on difficult conditions:** Motion blur, glare, and perspective distortion degrade accuracy — a CNN-based OCR model would generalise better
- **No ensemble/voting:** Single OCR pass without confidence scoring; majority voting across preprocessing variants would improve reliability
- **Cyrillic/non-Latin plates:** Current character whitelist covers Latin + Nordic characters only

## 🚀 Future Work

- [ ] Replace Tesseract with a custom CNN-LSTM OCR model trained on European plate fonts
- [ ] Add automated perspective correction before OCR for angled plates
- [ ] Extend character whitelist for Cyrillic plates (Bulgaria, Russia)
- [ ] Implement majority voting across multiple preprocessing outputs
- [ ] Add real-time video feed support

---

## 🎓 Academic Context

Developed as a First-Class BSc Computing Science dissertation at the **University of East Anglia** (module CMPP6001Y), supervised by Taoyang Wu. Full dissertation available on request.

---

## 👩‍💻 Author

**Seerat Kaur** — Junior Data Analyst | Python · SQL · Power BI · Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/seerat-kaur-4878bb249)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/KaurSeerat)
