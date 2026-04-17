import torch
import cv2
import numpy as np
import argparse
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.transforms import Resize, PrepareForNet 

# Load MiDaS model
def load_midas_model(model_path="src/model-f6b98070.pt"):
    try:
        model = MidasNet(model_path, non_negative=True)
        model.eval()
        model_transform = Compose([
            Resize(384, 384),
            PrepareForNet()
        ])
        return model, model_transform
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        return None, None

# Load YOLOv5 model
def load_yolo_model(model_path="srcyolov5s.pt"):
    try:
        return torch.hub.load("ultralytics/yolov5", "custom", path=model_path)
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return None


def process_midas_image(input_image, transform):

    if input_image.shape[-1] == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    transformed_image = transform({"image": input_image})["image"]

    input_tensor = torch.from_numpy(transformed_image).float()
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
    return input_tensor

def estimate_depth(image, model, transform):
    input_tensor = process_midas_image(image, transform)
    with torch.no_grad():
        prediction = model(input_tensor)
        depth_map = prediction.squeeze().cpu().numpy()
    return depth_map

def normalize_depth(depth_map):
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
    return normalized_depth

def detect_objects(image, model, selected_classes=None):

    results = model(image)
    detections = results.xyxy[0].cpu().numpy() 

    if selected_classes:
        detections = [box for box in detections if int(box[5]) in selected_classes]

    return detections

def draw_results(image, depth_map, boxes, focal_length=1.0): 
    depth_map = normalize_depth(depth_map) 
    for box in boxes:
        x1, y1, x2, y2, conf, cls = map(int, box)

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        if center_y >= depth_map.shape[0] or center_x >= depth_map.shape[1]:
            continue 

        depth = depth_map[center_y, center_x] 


        width_pixels = x2 - x1
        object_width = width_pixels / (focal_length * max(depth, 0.1)) 

        label = f"Depth: {depth:.2f}m"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(label)
    return image


def main(video_path=None, midas_model_path="src/model-f6b98070.pt", yolo_model_path="src/yolov5s.pt", selected_classes=None):
    midas_model, midas_transform = load_midas_model(midas_model_path)
    yolo_model = load_yolo_model(yolo_model_path)

    if midas_model is None or yolo_model is None:
        print("Failed to load models. Exiting...")
        return
    
    cap = cv2.VideoCapture(video_path if video_path else 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        depth_map = estimate_depth(frame, midas_model, midas_transform)

        detections = detect_objects(frame, yolo_model, selected_classes=selected_classes)

        output_frame = draw_results(frame, depth_map, detections)

        cv2.imshow("Distance Estimation", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distance estimation using MiDaS and YOLOv5")
    parser.add_argument("--video_path", type=str, help="Path to the video file")
    parser.add_argument("--midas_model_path", type=str, default="model-f6b98070.pt", help="Path to the MiDaS model")
    parser.add_argument("--yolo_model_path", type=str, default="yolov5s.pt", help="Path to the YOLOv5 model")
    parser.add_argument("--selected_classes", type=str, default="person,car", help="Comma-separated class names to detect (e.g., person,car)")
    args = parser.parse_args()

    class_names = args.selected_classes.split(",")
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
        "stop sign":11,
    }


    selected_classes = [class_map[name] for name in class_names if name in class_map]

    main(args.video_path, args.midas_model_path, args.yolo_model_path, selected_classes)