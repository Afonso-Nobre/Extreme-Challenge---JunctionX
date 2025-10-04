import os

from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from numpy.matlib import empty
from werkzeug.utils import secure_filename

app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "mp3",
    "mp4",
    "m4a", }

def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    message_type = "info"
    uploaded_info = None

    if request.method == 'POST':
        file = request.files.get("file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            if len(os.listdir(UPLOAD_FOLDER)) > 0 :
                for f in os.listdir(UPLOAD_FOLDER):
                    os.remove(os.path.join(UPLOAD_FOLDER, f))
            file.save(filepath)

            # App’s generated feedback (placeholder for now)
            feedback = "File processed successfully."

            uploaded_info = {
                "filename": filename,
                "feedback": feedback
            }

            message = "file uploaded successfully."
            message_type = "success"
        else:
            message = "Invalid file format."
            message_type = "error"

    return render_template('index.html', uploaded_info=uploaded_info, message = message, message_type = message_type)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == '__main__':
    app.run(debug=True)