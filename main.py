import cv2
import time
import math
from tracker import Tracker
from ultralytics import YOLO

# -----------------------------
# Load YOLOv8 model (auto-download)
# -----------------------------
model = YOLO("yolov8n.pt")

# Initialize tracker
tracker = Tracker()

# -----------------------------
# Video input
# -----------------------------
cap = cv2.VideoCapture("highway.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

# Store previous positions and time
prev_positions = {}
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, conf=0.5)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            detections.append([x1, y1, w, h])

    # Update tracker
    tracked_objects = tracker.update(detections)
    current_time = time.time()

    for x, y, w, h, obj_id in tracked_objects:
        # Center of bounding box
        cx = x + w // 2
        cy = y + h // 2

        speed = 0.0
        if obj_id in prev_positions:
            # Pixel distance
            dist = math.hypot(
                cx - prev_positions[obj_id][0],
                cy - prev_positions[obj_id][1]
            )

            time_diff = current_time - prev_time
            if time_diff > 0:
                speed = dist / time_diff  # pixel/sec

        prev_positions[obj_id] = (cx, cy)

        # Draw bounding box and speed
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{obj_id} Speed:{speed:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    prev_time = current_time

    cv2.imshow("Object Speed Estimation", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
