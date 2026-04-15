import cv2
import os

# Load cascade safely
cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error loading cascade file")

# ---------- REGISTER FACE ----------
def register_face(name):
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Camera failed to open")
        return

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Frame not captured")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            os.makedirs("known_faces", exist_ok=True)
            cv2.imwrite(f"known_faces/{name}.jpg", face)

            cap.release()
            cv2.destroyAllWindows()
            return

        cv2.imshow("Register Face", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- RECOGNIZE FACE ----------
def recognize_face():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Camera failed to open")
        return None

    known_faces = {}

    # Load saved faces
    for file in os.listdir("known_faces"):
        path = os.path.join("known_faces", file)
        img = cv2.imread(path, 0)
        if img is not None:
            known_faces[file.split(".")[0]] = img

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Frame not captured")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            for name, known in known_faces.items():
                try:
                    known_resized = cv2.resize(known, (w, h))
                except:
                    continue

                diff = cv2.absdiff(known_resized, face)
                score = diff.mean()

                if score < 50:
                    cap.release()
                    cv2.destroyAllWindows()
                    return name

        cv2.imshow("Recognize", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
