"""Face‑recognition attendance system
Pure‑Python / NumPy implementation with a *trained* centroid‑based classifier.

Features
========
• Multiple samples per user → mean vector stored in SQLite.  
• Centroid + cosine‑similarity classifier trained once, kept in RAM.  
• Live attendance stream **and** per‑user verification (search) stream.  
• Automatic model rebuild after every registration.
"""

from __future__ import annotations

import cv2
import numpy as np
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

# ───────────────────────────── configuration ────────────────────────────────── #

DB_PATH = Path("users.db")
SIM_THRESH = 0.75     # cosine‑similarity cut‑off for a match (tune this!)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

app = Flask(__name__)
app.secret_key = "📸secret"

# ───────────────────────────── global model ─────────────────────────────────── #

CENTROIDS: np.ndarray | None = None   # shape (K, 1024)
ID_ORDER: List[int] = []              # DB ids parallel to CENTROIDS rows

# ───────────────────────────── DB helpers ───────────────────────────────────── #


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                   id       INTEGER PRIMARY KEY AUTOINCREMENT,
                   name     TEXT UNIQUE,
                   encoding BLOB)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS attendance (
                   id       INTEGER PRIMARY KEY AUTOINCREMENT,
                   user_id  INTEGER,
                   date     TEXT,
                   time     TEXT,
                   UNIQUE(user_id, date))"""
        )
        conn.commit()


def save_user(name: str, vec: np.ndarray) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO users(name,encoding) VALUES(?,?)", (name, vec.tobytes()))
        conn.commit()


def get_users() -> List[Tuple[int, str]]:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT id,name FROM users").fetchall()


def get_user_encoding(uid: int) -> np.ndarray | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT encoding FROM users WHERE id=?", (uid,)).fetchone()
    return np.frombuffer(row[0], dtype="float32") if row else None


def all_user_encodings() -> List[Tuple[int, str, np.ndarray]]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id,name,encoding FROM users").fetchall()
    return [(r[0], r[1], np.frombuffer(r[2], dtype="float32")) for r in rows]


# attendance helpers ----------------------------------------------------------- #

present_users: set[int] = set()


def mark_attendance(uid: int):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        try:
            conn.execute(
                "INSERT INTO attendance(user_id,date,time) VALUES(?,?,?)",
                (uid, today, now),
            )
            present_users.add(uid)
        except sqlite3.IntegrityError:
            present_users.add(uid)


def get_present_users_today() -> List[str]:
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        rows = conn.execute(
            """SELECT DISTINCT users.name
                   FROM attendance
                   JOIN users ON attendance.user_id = users.id
                   WHERE attendance.date = ?""",
            (today,),
        ).fetchall()
    return [row[0] for row in rows]

# ───────────────────────── face utilities ───────────────────────────────────── #

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def vectorise_face(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    sobelx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx ** 2 + sobely ** 2)
    vec = edges.flatten().astype("float32")
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm

# ───────────────────────── model training & inference ───────────────────────── #


def load_centroid_matrix() -> Tuple[np.ndarray, List[int]]:
    rows = all_user_encodings()
    if not rows:
        return np.empty((0, 1024), dtype="float32"), []
    C = np.stack([r[2] for r in rows]).astype("float32")
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    ids = [r[0] for r in rows]
    return C, ids


def rebuild_model():
    global CENTROIDS, ID_ORDER
    CENTROIDS, ID_ORDER = load_centroid_matrix()


rebuild_model()


def predict_user(vec: np.ndarray) -> int | None:
    if CENTROIDS.size == 0:
        return None
    sims = CENTROIDS @ vec
    k = int(np.argmax(sims))
    return ID_ORDER[k] if sims[k] > SIM_THRESH else None

# ───────────────────────── helpers for MJPEG ---------------------------------- #


def encode_frame(frame):
    ret, buf = cv2.imencode(".jpg", frame)
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

# ───────────────────────── sample capture ------------------------------------- #


def capture_multiple_vectors(n_samples: int = 5) -> List[np.ndarray]:
    cap = cv2.VideoCapture(0)
    collected: List[np.ndarray] = []
    while len(collected) < n_samples and cap.isOpened():
        ret, frame = cap.read();    
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(
            frame,
            f"Samples {len(collected)}/{n_samples} | 'c' to capture | 'q' quit",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.imshow("Register", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces):
            x,y,w,h = faces[0]
            roi = frame[y:y+h, x:x+w]
            collected.append(vectorise_face(roi))
        if key == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()
    return collected

# ───────────────────────── camera streams ------------------------------------- #


def gen_attendance_stream():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read();  
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            roi = frame[y:y+h, x:x+w]
            uid = predict_user(vectorise_face(roi))
            if uid is not None:
                name = next(n for n_id, n in get_users() if n_id == uid)
                mark_attendance(uid)
                color,label = (0,255,0), name
            else:
                color,label = (0,0,255), "Unknown"
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        yield encode_frame(frame)
    cap.release()


def gen_search_stream(ref_vec: np.ndarray):
    # ref_vec is already normalized, don't normalize again
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            vec = vectorise_face(frame[y:y+h, x:x+w])
            sim = float(np.dot(ref_vec, vec))
            match = sim > SIM_THRESH
            color = (0, 255, 0) if match else (0, 0, 255)
            label = "MATCH" if match else f"{sim:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        yield encode_frame(frame)
    cap.release()

# ───────────────────────── Flask routes --------------------------------------- #

@app.route("/")
def home():
    return render_template("index.html", users=get_users())


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Name is required")
            return redirect(url_for("register"))

        vectors = capture_multiple_vectors()
        if not vectors:
            flash("Face capture failed, try again")
            return redirect(url_for("register"))
        vec_mean = np.mean(vectors, axis=0)
        vec_mean /= np.linalg.norm(vec_mean) + 1e-8
        vec_mean = vec_mean.astype("float32")

        try:
            save_user(name, vec_mean)
            flash(f"✅ {name} registered")
            rebuild_model()
        except sqlite3.IntegrityError:
            flash("Name already exists")
        return redirect(url_for("home"))
    return render_template("register.html")


@app.route("/camera_feed/<int:uid>")
def camera_feed(uid):
    ref_vec = get_user_encoding(uid)
    if ref_vec is None:
        return "Unknown user", 404
    ref_vec = ref_vec.astype("float32", copy=True)  # make writable copy
    return Response(gen_search_stream(ref_vec), mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/attendance/run")
def attendance_run():
    return Response(gen_attendance_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/attendance")
def attendance_page():
    names = get_present_users_today()
    return render_template("attendance.html", present_names=names)


@app.route("/api/present-users")
def api_present_users():
    names = get_present_users_today()
    return {"present": names}


if __name__ == "__main__":
    init_db()
    app.run(debug=True, threaded=True)
