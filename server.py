from flask import Flask, render_template, request
import firebase_admin
from firebase_admin import storage

app = Flask(__name__)

# Initialize Firebase (use Render's secret file)
cred = firebase_admin.credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred, {"storageBucket": "exam-surveillance.firebasestorage.app"})

@app.route('/')
def upload_page():
    return render_template('upload.html')  # Serve your HTML

@app.route('/upload', methods=['POST'])
def handle_upload():
    video = request.files['video']
    blob = storage.bucket().blob(f"videos/{video.filename}")
    blob.upload_from_file(video)
    return "Upload successful! Analysis will begin shortly."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)