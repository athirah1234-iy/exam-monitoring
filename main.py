import os
import cv2
import requests
import shutil
import firebase_admin
from firebase_admin import credentials, storage
from tqdm import tqdm
from ultralytics import YOLO
import mediapipe as mp

# ========= CONFIG ========= #
FIREBASE_BUCKET = "exam-surveillance.firebasestorage.app"  # 🚀 Replace with your bucket
TELEGRAM_BOT_TOKEN = "7556496216:AAEnklt80gw-kF7ZHFRA2o7D91YDKCeXMNU"  # 🚀 Replace
TELEGRAM_CHAT_ID = "1042670358"  # 🚀 Replace
TEMP_FOLDER = "TEMP_DOWNLOADS"
OUTPUT_DIR = "AI_Cheating_Reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ========= MODEL DOWNLOADER ========= #
def download_model(bucket, remote_path, local_dir="/tmp/models"):
    """Downloads files from Firebase Storage to /tmp/models/"""
    os.makedirs(local_dir, exist_ok=True)
    blob = bucket.blob(remote_path)
    local_path = os.path.join(local_dir, os.path.basename(remote_path))
    blob.download_to_filename(local_path)
    print(f"⬇️ Downloaded model: {remote_path} → {local_path}")
    return local_path

# ========= INIT FIREBASE ========= #
def initialize_firebase():
    try:
        firebase_admin.get_app()
    except ValueError:
        # 🚨 Remove FIREBASE_CREDENTIALS_FILE! Render uses /etc/secrets/
        cred = credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        firebase_admin.initialize_app(cred, {'storageBucket': FIREBASE_BUCKET})
    return storage.bucket()

# ========= INIT MODELS ========= #
bucket = initialize_firebase()
YOLO_MODEL_PATH = download_model(bucket, "models/yolov8n-pose.pt")  # 🚀 Path in Firebase
pose_model = YOLO(YOLO_MODEL_PATH)

# Download DNN models
DNN_MODEL_PATH = download_model(bucket, "models/frozen_inference_graph.pb")  # 🚀 Replace
DNN_CONFIG_PATH = download_model(bucket, "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")  # 🚀 Replace
cv_net = cv2.dnn.readNetFromTensorflow(DNN_MODEL_PATH, DNN_CONFIG_PATH)

# ========= DOWNLOAD FROM FIREBASE ========= #
def download_videos(bucket):
    blobs = bucket.list_blobs(prefix="videos/")  # ✅ Your bucket uses 'videos/', not 'video/'
    video_files = []
    for blob in blobs:
        if blob.name.endswith(('.mp4', '.mov', '.avi')):
            local_path = os.path.join(TEMP_FOLDER, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            video_files.append(os.path.basename(blob.name))
            print(f"⬇️ Downloaded: {blob.name}")
    return video_files

# ========= DETECT CHEATING ========= #
def detect_cheating(video_path):
    print(f"📹 Analyzing: {video_path}")
    alerts = set()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = 0
    consecutive_multi_person_frames = 0
    MULTI_PERSON_THRESHOLD = 5  # Raise alert only if multiple people appear for 5+ frames

    with mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2) as hand_detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"🧠 Frame {frame_count}")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ========= People Detection =========
            pose_results = pose_model(frame, verbose=False)
            pose_data = pose_results[0] if isinstance(pose_results, (list, tuple)) else pose_results
            people_count = len(pose_data.boxes) if pose_data.boxes is not None else 0

            if people_count > 1:
                consecutive_multi_person_frames += 1
            else:
                consecutive_multi_person_frames = 0

            if consecutive_multi_person_frames == MULTI_PERSON_THRESHOLD:
                alerts.add("🚨 Multiple people consistently detected")

            # ========= Phone Detection =========
            hand_results = hand_detector.process(rgb_frame)
            if hasattr(hand_results, 'multi_hand_landmarks') and hand_results.multi_hand_landmarks:
                blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True)
                cv_net.setInput(blob)
                detections = cv_net.forward()
                for detection in detections[0][0]:
                    if int(detection[1]) == PHONE_CLASS_ID and detection[2] > 0.7:
                        alerts.add("📱 Phone detected")
                        break

    cap.release()
    return list(alerts)

# ========= PROCESS SINGLE VIDEO ========= #
def process_video(video_file):
    video_path = os.path.join(TEMP_FOLDER, video_file)
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Missing video: {video_path}")
        alerts = detect_cheating(video_path)
        return {'video': video_file, 'alerts': alerts}
    except Exception as e:
        return {'video': video_file, 'error': str(e)}

# ========= MAIN ========= #
def main():
    try:
        bucket = initialize_firebase()
        videos = download_videos(bucket)

        if not videos:
            print("📂 No videos found in Firebase.")
            return

        results = []
        for video in tqdm(videos, desc="Analyzing"):
            results.append(process_video(video))

        critical_alerts = []
        for result in results:
            report_path = os.path.join(OUTPUT_DIR, f"{result['video']}_report.txt")

            if 'alerts' in result:
                alerts = result['alerts']
                content = "\n".join(alerts) if alerts else "✅ No cheating detected"
                with open(report_path, 'w') as f:
                    f.write(content)

                if alerts:
                    critical_alerts.append(f"{result['video']}:\n" + "\n".join(alerts))
            else:
                with open(report_path, 'w') as f:
                    f.write(f"❌ Error: {result['error']}")

        if critical_alerts:
            send_telegram("🚨 Critical Alerts:\n\n" + "\n\n".join(critical_alerts))
        else:
            print("✅ No critical alerts detected.")

    finally:
        shutil.rmtree(TEMP_FOLDER, ignore_errors=True)
        print("📁 Temporary folder cleaned up.")

if __name__ == "__main__":
    main()
