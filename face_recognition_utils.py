import face_recognition
import cv2
import pickle
import os

ENCODINGS_FILE = "encodings.pkl"

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}

def save_encodings(data):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

def register_face(name):
    video_capture = cv2.VideoCapture(0)

    print("Capturing face... Press 'q' to capture")

    while True:
        ret, frame = video_capture.read()
        cv2.imshow("Register Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rgb_frame = frame[:, :, ::-1]
            faces = face_recognition.face_locations(rgb_frame)

            if len(faces) > 0:
                encoding = face_recognition.face_encodings(rgb_frame, faces)[0]

                data = load_encodings()
                data["encodings"].append(encoding)
                data["names"].append(name)
                save_encodings(data)

                break

    video_capture.release()
    cv2.destroyAllWindows()