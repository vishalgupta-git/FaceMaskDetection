import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
yolo_face = YOLO("yolov8nfa.pt")
yolo_face.verbose = False

# Load mask classification CNN model
mask_model = load_model("face_mask_model.keras")


def preprocess(face_img):
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_face(face_crop):
    """
    Receives a cropped face and returns the classification label only.
    """
    processed = preprocess(face_crop)
    pred = mask_model.predict(processed)[0]

    if pred > 0.5:
        return "Without Mask", float(pred)
    else:
        return "With Mask", float(1 - pred)

def detect_and_annotate(frame):
    """
    Detects faces with YOLO, classifies each face using CNN,
    draws bounding boxes + labels, and returns annotated frame.
    """

    annotated = frame.copy()
    results = yolo_face(frame,verbose=False)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            cls_id = int(box.cls[0]) if box.cls is not None else 0
            yolo_label = yolo_face.names.get(cls_id, "Face")

            mask_label, conf = classify_face(face_crop)

            final_label = f"{yolo_label}: {mask_label} ({conf:.2f})"

            color = (0, 255, 0) if mask_label == "With Mask" else (0, 0, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

    return annotated
