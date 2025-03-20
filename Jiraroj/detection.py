import cv2
import torch  # ✅ Fix: Import torch
import numpy as np
from ultralytics import YOLO

# Mapping of A0...A43 to Thai characters (loaded from Excel)
label_to_thai_mapping = {
    'A0': 'ก', 'A1': 'ข', 'A2': 'ฃ', 'A3': 'ค', 'A4': 'ฅ',
    'A5': 'ฆ', 'A6': 'ง', 'A7': 'จ', 'A8': 'ฉ', 'A9': 'ช',
    'A10': 'ซ', 'A11': 'ฌ', 'A12': 'ญ', 'A13': 'ฎ', 'A14': 'ฏ',
    'A15': 'ฐ', 'A16': 'ฑ', 'A17': 'ฒ', 'A18': 'ณ', 'A19': 'ด',
    'A20': 'ต', 'A21': 'ถ', 'A22': 'ท', 'A23': 'ธ', 'A24': 'น',
    'A25': 'บ', 'A26': 'ป', 'A27': 'ผ', 'A28': 'ฝ', 'A29': 'พ',
    'A30': 'ฟ', 'A31': 'ภ', 'A32': 'ม', 'A33': 'ย', 'A34': 'ร',
    'A35': 'ฤ', 'A36': 'ล', 'A37': 'ว', 'A38': 'ศ', 'A39': 'ษ',
    'A40': 'ส', 'A41': 'ห', 'A42': 'ฬ', 'A43': 'ฮ'
}

def load_models(plate_model_path, content_model_path):
    plate_model = YOLO(plate_model_path)
    content_model = YOLO(content_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plate_model.to(device)
    content_model.to(device)
    return plate_model, content_model

def detect_license_plates(model, image, confidence_threshold=0.5):
    results = model(image, conf=confidence_threshold, half=torch.cuda.is_available())
    if len(results[0].boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    return boxes, confidences, class_ids

def detect_contents(model, plate_image, confidence_threshold=0.5):
    results = model(plate_image, conf=confidence_threshold, half=torch.cuda.is_available())
    if len(results[0].boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    return boxes, confidences, class_ids

def sort_contents_left_to_right(boxes, confidences, class_ids):
    contents = sorted(zip(boxes, confidences, class_ids), key=lambda x: x[0][0])
    sorted_boxes, sorted_confidences, sorted_class_ids = zip(*contents)
    return np.array(sorted_boxes), np.array(sorted_confidences), np.array(sorted_class_ids)

def main():
    plate_model_path = "/home/pi/Documents/best2.pt"
    content_model_path = "/home/pi/Documents/best.pt"
    plate_model, content_model = load_models(plate_model_path, content_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes, confidences, class_ids = detect_license_plates(plate_model, frame)
        if len(boxes) > 0:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                confidence = confidences[i]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"License Plate {confidence:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                plate = frame[int(y1):int(y2), int(x1):int(x2)]
                content_boxes, content_confidences, content_class_ids = detect_contents(content_model, plate)
                if len(content_boxes) > 0:
                    sorted_boxes, sorted_confidences, sorted_class_ids = sort_contents_left_to_right(
                        content_boxes, content_confidences, content_class_ids
                    )
                    content_text = ""
                    for j in range(len(sorted_boxes)):
                        cx1, cy1, cx2, cy2 = sorted_boxes[j]
                        content_confidence = sorted_confidences[j]
                        content_class_id = sorted_class_ids[j]
                        thai_char = label_to_thai_mapping.get(f"A{int(content_class_id)}", str(content_class_id))
                        content_text += thai_char
                        cv2.rectangle(plate, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (255, 0, 0), 2)
                        cv2.putText(plate, f"{thai_char} {content_confidence:.2f}", (int(cx1), int(cy1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(frame, f"Contents: {content_text}", (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
