from flask import Flask, render_template, request, redirect
import sqlite3
from datetime import datetime
from face_utils import register_face, recognize_face

app = Flask(__name__)

# ---------- DATABASE ----------
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

    # prevent duplicate attendance
    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
    if not cursor.fetchone():
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
    name = recognize_face()
    if name:
        mark_attendance(name)
        return f"{name} marked present!"
    return "Face not recognized"

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
