from flask import Flask, render_template, request, redirect
import sqlite3
from datetime import datetime
import cv2
import face_recognition
from face_recognition_utils import register_face, load_encodings

app = Flask(__name__)

# ---------- DATABASE INIT ----------
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            time TEXT
        )
    """)

    conn.commit()
    conn.close()

# ---------- MARK ATTENDANCE ----------
def mark_attendance(name):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # prevent duplicate for same day
    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    already_marked = cursor.fetchone()

    if not already_marked:
        cursor.execute(
            "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
            (name, date, time)
        )

    conn.commit()
    conn.close()

# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    name = request.form["name"]
    register_face(name)
    return redirect("/")

@app.route("/mark")
def mark():
    data = load_encodings()
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]

        faces = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, faces)

        for face_encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)

            if True in matches:
                index = matches.index(True)
                name = data["names"][index]

                mark_attendance(name)

                video_capture.release()
                cv2.destroyAllWindows()
                return f"{name} marked present!"

        cv2.imshow("Mark Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return "No face recognized"

@app.route("/attendance")
def attendance():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM attendance")
    data = cursor.fetchall()

    conn.close()
    return render_template("attendance.html", data=data)

# ---------- MAIN ----------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)