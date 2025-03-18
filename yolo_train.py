import os
import wandb
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd

# Set environment variable to avoid OpenMP runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
# Initialize WandB
wandb.init(
    project="yolo-training",
    name="eu_plate_model",
    config={
        "epochs": 10,
        "batch_size": 16,
        "img_size":640,
        "learning_rate": 0.01,
    },
)
# Define fixed save directory to avoid multiple folders
project_dir = "./yolo-training"
run_name = "eu_plate_model"  
save_dir = os.path.join(project_dir, run_name)

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Path to dataset YAML file
dataset_path = r"./dataset/"  
data_yaml = os.path.join(dataset_path, "data.yaml")

# Initialize YOLO model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Train YOLO model
train_results = model.train(
    data=data_yaml,  # Path to dataset YAML
    epochs=10,
    imgsz=640,
    batch=16,
    name="eu_plate_model",  # Training run name
    project="yolo-training",  # Project name for WandB
    verbose=True,  # Print logs during training
    save_period=1  # Save weights after each epoch
)

# Validate YOLO model on validation set and store in same directory
val_results = model.val(save_dir=save_dir)
print("Validation results:", val_results)

# Plot Training Metrics from results.csv
results_csv_path = os.path.join(save_dir, "results.csv")

if os.path.exists(results_csv_path):
    df = pd.read_csv(results_csv_path)
    print("Available columns in results.csv:", df.columns)

    # Plot Training & Validation Losses
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    plt.plot(df["epoch"], df["val/box_loss"], label="Validation Box Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train & Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "train_val_loss.png"))
    plt.show()

    # Check if mAP columns exist before plotting
    map_50_col = "metrics/mAP_50" if "metrics/mAP_50" in df.columns else "metrics/mAP_50(B)"
    map_50_95_col = "metrics/mAP_50-95" if "metrics/mAP_50-95" in df.columns else "metrics/mAP_50-95(B)"

    if map_50_col in df.columns and map_50_95_col in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["epoch"], df[map_50_col], label="mAP@50")
        plt.plot(df["epoch"], df[map_50_95_col], label="mAP@50-95")
        plt.xlabel("Epochs")
        plt.ylabel("Mean Average Precision")
        plt.title("mAP@50 and mAP@50-95 Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "map_performance.png"))
        plt.show()
    else:
        print("Warning: mAP columns not found in results.csv. Skipping mAP plot.")


# Validate YOLO model
val_results = model.val()
print("Validation results:", val_results)

# Run inference on test images
test_images_path = os.path.join(dataset_path, "test", "images")
results = model.predict(source=test_images_path, save=True, conf=0.5)
print(f"Predictions saved in: {save_dir}")

# Evaluate on test set
test_results = model.val(data=data_yaml, split="test")
print("Test results:", test_results)

# Ensure loss and precision plots are stored in the same directory
plot_save_path = os.path.join(save_dir, "plots")
os.makedirs(plot_save_path, exist_ok=True)