from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import face_recognition
import pickle
import numpy as np
import time
from database import init_db, get_users, save_user, get_user_encoding

app = Flask(__name__)
VIDEO_PATHS = ['videos/video1.mp4', 'videos/video2.mp4']

init_db()

@app.route('/')
def index():
    users = get_users()
    return render_template('index.html', users=users)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        encoding = capture_face()
        if encoding is None:
            return "❗ No face captured. Please try again.", 400
        save_user(name, encoding)
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/video_feed/<int:user_id>/<int:video_num>')
def video_feed(user_id, video_num):
    print(f"▶️ video_feed called for user {user_id}, video {video_num}")
    encoding = get_user_encoding(user_id)
    return Response(generate_video(VIDEO_PATHS[video_num], encoding),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_face():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    encoding = None
    print("📸 Align your face and press 'q' when ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        locations = face_recognition.face_locations(rgb)

        for top, right, bottom, left in locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if locations:
            cv2.putText(frame, "Face Detected — Press 'q'", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            if locations:
                enc = face_recognition.face_encodings(rgb, locations)
                if enc:
                    encoding = enc[0]
                    print("✅ Face captured.")
                    break
                else:
                    print("⚠️ Encoding failed. Try again.")
            else:
                print("❗ No face detected. Press 'q' only when a face is visible.")

    cap.release()
    cv2.destroyAllWindows()
    return encoding

def generate_video(video_path, target_encoding):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if FPS is unknown
    timer_start = time.time()
    timeout = 5  # Timeout in seconds
    frame_count = 0
    skip_frames = 3  # Process every 3rd frame

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        start_time = time.time()
        locations = face_recognition.face_locations(rgb, model="hog")
        print(f"🔍 Detected {len(locations)} faces in frame (detection took {time.time() - start_time:.2f}s)")
        encodings = face_recognition.face_encodings(rgb, locations) if locations else []

        match_found = False
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            if target_encoding is not None and face_recognition.compare_faces([target_encoding], encoding, tolerance=0.5)[0]:
                match_found = True
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Found", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add "Found" text

        if match_found:
            timer_start = time.time()  # Reset timer on match
        elif time.time() - timer_start > timeout:
            cv2.putText(frame, "Not Found", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    video.release()

if __name__ == "__main__":
    app.run(debug=True, threaded=True)






























    