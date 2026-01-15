import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

print("Loading YOLO and Keras models...")
yolo_face = YOLO("yolov8nfa.pt")
mask_model = load_model("face_mask_model.keras")
print("Models loaded successfully.")

def preprocess_face(face_img):
    """
    Preprocesses a single face crop for the CNN model.
    """
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256)) 
    img = img / 255.0
    return img

def detect_and_annotate(frame):
    annotated = frame.copy()
    
    results = yolo_face(frame, verbose=False, imgsz=640)
    
    faces_to_predict = []
    face_coords = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            processed_face = preprocess_face(face_crop)
            faces_to_predict.append(processed_face)
            face_coords.append((x1, y1, x2, y2))

    if len(faces_to_predict) > 0:
        batch_input = np.array(faces_to_predict)
        predictions = mask_model.predict(batch_input, verbose=0)

        for i, (x1, y1, x2, y2) in enumerate(face_coords):
            pred = predictions[i][0] 
            
            if pred > 0.5:
                label = "No Mask"
                conf = float(pred)
                color = (0, 0, 255) # Red
            else:
                label = "Mask"
                conf = float(1 - pred)
                color = (0, 255, 0) # Green

            text_label = f"{label}: {conf:.2f}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            
            # cv2.putText(annotated, text_label, (x1, y1 - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return annotated