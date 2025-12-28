from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = {"car","bus","truck","motorcycle"}

def detect_vehicles(image, conf_threshold=0.5,show_boxes=True):
    if not isinstance(image,np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    
    results = model(image,
                    conf = conf_threshold,
                    verbose = False)
    
    counts = {
        "car": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0
    }

    annotated_image = image.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]

        if label not in VEHICLE_CLASSES:
            continue

        counts[label] += 1

        if show_boxes:
            # Ensure coordinates are standard Python ints (not numpy types or iterators)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            confidence = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0

            # Draw rectangle and label; ensure org is a tuple of two ints
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            text = f"{label} {confidence:.2f}"
            org = (int(x1), int(max(y1 - 10, 0)))

            cv2.putText(
                annotated_image,
                str(text),
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    return annotated_image, counts