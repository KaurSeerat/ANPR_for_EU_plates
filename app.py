from flask import Flask, render_template, request, redirect, send_from_directory
import os
from werkzeug.utils import secure_filename
from ocr import process_image

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            print(f"File saved: {file_path}")

            # Process the image and store it in the results folder
            processed_image_path, detected_texts = process_image(file_path, app.config['PROCESSED_FOLDER'])

            if not detected_texts:
                print("No plates detected.")

            # Extract only the filename for correct routing
            processed_image_filename = os.path.basename(processed_image_path)

            return render_template(
                "index.html",
                uploaded_file=filename,
                processed_file=processed_image_filename,  
                detected_texts=detected_texts
            )

    return render_template("index.html", uploaded_file=None, processed_file=None, detected_texts=[])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
