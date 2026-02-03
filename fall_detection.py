import cv2
import mediapipe as mp
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

fall_start = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        shoulder_y = lm[11].y * h
        hip_y = lm[23].y * h

        diff = abs(shoulder_y - hip_y)

        if diff < 40:
            if fall_start is None:
                fall_start = time.time()
            elif time.time() - fall_start > 1.5:
                cv2.putText(frame, "FALL DETECTED",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0,0,255), 4)
        else:
            fall_start = None

    cv2.imshow("VisionGuard - Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
