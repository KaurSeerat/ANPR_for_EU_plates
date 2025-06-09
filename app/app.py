"""
File: app.py
Description: 
This is the main Flask backend for web application.
"""
from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import os
from werkzeug.utils import secure_filename
import sys

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from data_preprocessing.ocr import process_image
app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'results')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'PROCESSED_FOLDER': PROCESSED_FOLDER,
    'STATIC_FOLDER': STATIC_FOLDER,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024  # 16MB max upload
})


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Process the image
                processed_image_path, detected_texts = process_image(
                    file_path, 
                    app.config['PROCESSED_FOLDER']
                )

                # Get relative paths for web display
                uploaded_file_url = url_for('uploaded_file', filename=filename)
                processed_file_url = url_for('processed_file', filename=os.path.basename(processed_image_path))

                return render_template(
                    "index.html",
                    uploaded_file=filename,
                    uploaded_file_url=uploaded_file_url,
                    processed_file=os.path.basename(processed_image_path),
                    processed_file_url=processed_file_url,
                    detected_texts=detected_texts
                )

            except Exception as e:
                app.logger.error(f"Error processing file: {str(e)}")
                return render_template("index.html", error=str(e))

    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)