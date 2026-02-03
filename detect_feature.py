import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        shoulder = [lm[11].x * w, lm[11].y * h]
        hip = [lm[23].x * w, lm[23].y * h]

        vertical_dist = abs(shoulder[1] - hip[1])

        cv2.putText(frame, f"Vertical Dist: {int(vertical_dist)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("VisionGuard - Feature Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
