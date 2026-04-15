import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def register_face(name):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
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


def recognize_face():
    cap = cv2.VideoCapture(0)

    known_faces = {}
    for file in os.listdir("known_faces"):
        img = cv2.imread(f"known_faces/{file}", 0)
        known_faces[file.split(".")[0]] = img

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            for name, known in known_faces.items():
                known_resized = cv2.resize(known, (w, h))
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
