# -*- coding: utf-8 -*-
"""AI Attendance + Search System (no external ML libs)
Author: ChatGPT helper
Updated: 2025‑06‑28 — single‑shot registration (press **c** once).
"""

from __future__ import annotations

import cv2
import numpy as np
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Tuple

from flask import (
    Flask,
    Response,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

DB_PATH = Path("users.db")
THRESH = 0.60  # tunable Euclidean distance threshold
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

app = Flask(__name__)
app.secret_key = "📸secret"

# ───────────────────────────── DB helpers ────────────────────────────── #

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


def mark_attendance(uid: int):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        try:
            conn.execute("INSERT INTO attendance(user_id,date,time) VALUES(?,?,?)", (uid, today, now))
        except sqlite3.IntegrityError:
            pass

init_db()

# ───────────────────────── Face utilities ──────────────────────────── #

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def vectorise_face(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (16, 8), interpolation=cv2.INTER_AREA)  # 128‑pix
    vec = thumb.flatten().astype("float32")
    norm = np.linalg.norm(vec) or 1.0
    return vec / norm


def closest_user(vec: np.ndarray, encs):
    best_dist = float("inf")
    best = (None, None)
    for uid, name, ref in encs:
        dist = np.linalg.norm(vec - ref)
        if dist < best_dist:
            best_dist, best = dist, (uid, name)
    return (*best, best_dist) if best_dist < THRESH else (None, None, best_dist)

present_users = set()

def mark_attendance(uid: int):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        try:
            conn.execute("INSERT INTO attendance(user_id,date,time) VALUES(?,?,?)", (uid, today, now))
            present_users.add(uid)
        except sqlite3.IntegrityError:
            present_users.add(uid)

def get_present_users_today() -> List[str]:
    today = datetime.now().strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        rows = conn.execute("""
            SELECT DISTINCT users.name
            FROM attendance
            JOIN users ON attendance.user_id = users.id
            WHERE attendance.date = ?
        """, (today,)).fetchall()
    return [row[0] for row in rows]



# ─────────────────────────── Routes ──────────────────────────────── #

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
        vec = capture_single_vector()
        if vec is None:
            flash("Face capture failed, try again")
            return redirect(url_for("register"))
        try:
            save_user(name, vec)
            flash(f"✅ {name} registered")
        except sqlite3.IntegrityError:
            flash("Name already exists")
        return redirect(url_for("home"))
    return render_template("register.html")


@app.route("/camera_feed/<int:uid>")
def camera_feed(uid):
    ref_vec = get_user_encoding(uid)
    if ref_vec is None:
        return "Unknown user", 404
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



# ───────────────────────── Generators ────────────────────────────── #

def encode_frame(frame):
    ret, buf = cv2.imencode(".jpg", frame)
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


def capture_single_vector() -> np.ndarray | None:
    cap = cv2.VideoCapture(0)
    collected: List[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Press 'c' to capture | 'q' to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Register", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces):
            (x, y, w, h) = faces[0]
            roi = frame[y:y+h, x:x+w]
            collected.append(vectorise_face(roi))
            break
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return collected[0] if collected else None


def gen_search_stream(ref_vec):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            vec = vectorise_face(roi)
            dist = np.linalg.norm(vec - ref_vec)
            match = dist < THRESH
            color = (0,255,0) if match else (0,0,255)
            label = "MATCH" if match else f"{dist:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        yield encode_frame(frame)
    cap.release()


def gen_attendance_stream():
    encs = all_user_encodings()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            vec = vectorise_face(roi)
            uid, name, dist = closest_user(vec, encs)
            if uid is not None:
                mark_attendance(uid)
                color, label = (0,255,0), name
            else:
                color, label = (0,0,255), "Unknown"
            cv2.rectangle(frame, (x,y), (x+w, y+h), color,2)
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,2)
        yield encode_frame(frame)
    cap.release()

# ─────────────────────────── entry ──────────────────────────────── #

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
