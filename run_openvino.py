import cv2
import numpy as np
from openvino.runtime import Core
from ultralytics import YOLO

def load_midas_model(model_path="src/openvino_midas_v21_small.xml"):
    try:
        core = Core()
        compiled_model = core.compile_model(model_path, "CPU")
        input_layer = compiled_model.inputs[0]
        output_layer = compiled_model.outputs[0]
        return compiled_model, input_layer, output_layer
    except Exception as e:
        print(f"Error loading OpenVINO MiDaS model: {e}")
        return None, None, None

def process_midas_image(input_image, target_size=(256, 256)):
    if input_image.shape[-1] == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(input_image, target_size)
    input_image = input_image.astype(np.float32) / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

def estimate_depth(image, model, input_layer, output_layer):
    input_blob = process_midas_image(image)
    input_blob = np.ascontiguousarray(input_blob)
    result = model([input_blob])
    depth_map = result[output_layer].squeeze()
    return depth_map

def normalize_depth(depth_map):
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
    return (normalized_depth * 255).astype(np.uint8)

def calculate_object_distance(box, depth_map, scaling_factor):
    x1, y1, x2, y2 = map(int, box)
    object_depth = depth_map[y1:y2, x1:x2].copy()
    depth_map[y1:y2, x1:x2] = 0
    avg_depth = np.mean(object_depth)
    distance_cm = avg_depth * scaling_factor
    return distance_cm

def draw_detections(frame, detections, classes, confidences, depth_map, scaling_factor, class_names):
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(classes[i])
        confidence = confidences[i]
        class_name = class_names[class_id]
        distance_cm = calculate_object_distance(box, depth_map, scaling_factor)
        label = f"{class_name} ({confidence:.2f}), Dist: {distance_cm:.2f} (Relative Units)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main(midas_model_path="src/openvino_midas_v21_small.xml", yolo_model_path="src/yolov5s.pt", selected_classes=None):
    midas_model, input_layer, output_layer = load_midas_model(midas_model_path)
    if midas_model is None:
        print("Failed to load MiDaS model. Exiting...")
        return
    yolo_model = YOLO(yolo_model_path)
    class_names = yolo_model.names
    scaling_factor = 10/31
    class_map = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "parking meter": 4,
        "bus": 5,
        "train": 6,
        "truck": 7,
        "bench": 8,
        "traffic light": 9,
        "fire hydrant": 10,
        "stop sign": 11,
    }
    if selected_classes:
        class_names_filtered = selected_classes.split(",")
        selected_class_ids = [list(class_names.values()).index(class_name.strip()) for class_name in class_names_filtered if class_name.strip() in class_names.values()]
    else:
        selected_class_ids = list(class_names.keys())
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        depth_map = estimate_depth(frame, midas_model, input_layer, output_layer)
        depth_map_normalized = normalize_depth(depth_map)
        depth_map_copy = depth_map.copy()
        results = yolo_model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        if selected_classes:
            selected_indices = [i for i, cls_id in enumerate(classes) if cls_id in selected_class_ids]
            detections = detections[selected_indices]
            classes = classes[selected_indices]
            confidences = confidences[selected_indices]
        draw_detections(frame, detections, classes, confidences, depth_map_copy, scaling_factor, class_names)
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Depth Map", depth_map_normalized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time object distance estimation")
    parser.add_argument("--midas_model_path", type=str, default="openvino_midas_v21_small.xml", help="Path to MiDaS model (.xml file)")
    parser.add_argument("--yolo_model_path", type=str, default="yolov5s.pt", help="Path to YOLO model")
    parser.add_argument("--selected_classes", type=str, default=None, help="Comma-separated list of class names to detect (e.g., 'person,car')")
    args = parser.parse_args()
    main(args.midas_model_path, args.yolo_model_path, args.selected_classes)