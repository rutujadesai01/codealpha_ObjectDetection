import cv2
import torch
from ultralytics import YOLO  

# Load YOLOv8 model (smallest version for speed)
model = YOLO("yolov8n.pt").to('cuda' if torch.cuda.is_available() else 'cpu')

# Open webcam (or use video file)
cap = cv2.VideoCapture(0)

# Initialize tracking
tracker = cv2.TrackerCSRT_create() if hasattr(cv2, 'TrackerCSRT_create') else cv2.TrackerCSRT()
initBB = None

frame_count = 0
skip_frames = 2  # Process every 2nd frame for better performance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, (640, 480))  # Resize frame for speed

    if initBB is not None:
        # Update tracker
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Run YOLO detection on every `skip_frames`
        if frame_count % skip_frames == 0:
            results = model(frame)  # Run YOLO detection

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    label = model.names[int(box.cls[0])]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 0, 255), 2)

    cv2.imshow("Optimized YOLO Object Detection & Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Select ROI for tracking
        initBB = cv2.selectROI("Optimized YOLO Object Detection & Tracking", frame, fromCenter=False, showCrosshair=True)
        tracker.init(frame, initBB)
    elif key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
