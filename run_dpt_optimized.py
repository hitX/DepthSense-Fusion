import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from midas.dpt_depth import DPTDepthModel
from PIL import Image
import time
import cProfile


# Load PyTorch MiDaS model
def load_midas_model(model_type="dpt_hybrid", pretrained_path="src/dpt_hybrid_384.pt"):
    try:
        if model_type == "dpt_hybrid":
            model = DPTDepthModel(
                path=pretrained_path,
                backbone="vitb_rn50_384",
                non_negative=True
            )
        else:
            raise ValueError("Unsupported model type")

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        model.eval() 
        return model, device
    except Exception as e:
        print(f"Error loading PyTorch MiDaS model: {e}")
        return None, None


def process_midas_image(input_image, target_size=(384, 384), device=torch.device("cpu")):
    if input_image.shape[-1] == 3:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


    input_image = Image.fromarray(input_image)
    
    transforms = Compose([
            Resize(target_size, interpolation=Image.Resampling.BICUBIC),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_image = transforms(input_image)  
    input_image = input_image.to(device)
    input_image = input_image.unsqueeze(0)   
    return input_image




def estimate_depth(image, model, device):
    with torch.no_grad():
        depth_map = model(image).squeeze().cpu().numpy() 
    return depth_map


def normalize_depth(depth_map):
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
    return (normalized_depth * 255).astype(np.uint8)


def calculate_object_distance(box, depth_map, scaling_factor):
    x1, y1, x2, y2 = map(int, box)
    object_depth = depth_map[y1:y2, x1:x2]
    avg_depth = np.mean(object_depth)
    distance_cm = avg_depth * scaling_factor 
    return distance_cm


def main(midas_model_type="dpt_hybrid", midas_pretrained_path="src/dpt_hybrid_384.pt", yolo_model_path="src/yolov5s.pt", selected_classes=None, skip_frames = 1):

    midas_model, device = load_midas_model(midas_model_type, midas_pretrained_path)
    if midas_model is None:
        print("Failed to load MiDaS model. Exiting...")
        return


    yolo_model = YOLO(yolo_model_path)
    

    cpu_device = torch.device("cpu")


    scaling_factor = 10/31 

    # Class name to class ID mapping
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

    id_to_class_name = {v: k for k, v in class_map.items()}

    if selected_classes:
        class_names = selected_classes.split(",")
        selected_class_ids = [class_map[class_name.strip()] for class_name in class_names if class_name.strip() in class_map]
    else:
        selected_class_ids = list(class_map.values()) 

    cap = cv2.VideoCapture(0) 
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        frame_count += 1
        
        if frame_count % (skip_frames+1) != 0:
            continue


        start_time = time.time()

        resized_frame = process_midas_image(frame, device=device)


        depth_map = estimate_depth(resized_frame, midas_model, device)
        depth_map_normalized = normalize_depth(depth_map)
        
   
        results = yolo_model(frame)
        detections = results[0].boxes.xyxy.cpu().numpy()  
        classes = results[0].boxes.cls.cpu().numpy()      
        confidences = results[0].boxes.conf.cpu().numpy() 
        
 
        end_time = time.time()
        frame_time = end_time-start_time
        
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(classes[i])
            confidence = confidences[i]


            if class_id in selected_class_ids:

                class_name = id_to_class_name.get(class_id, "Unknown")


                distance_cm = calculate_object_distance(box, depth_map, scaling_factor)


                label = f"{class_name} ({confidence:.2f}), Dist: {distance_cm:.2f} cm"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

        fps = 1/frame_time
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        cv2.imshow("Original Frame", frame)
        cv2.imshow("Depth Map", depth_map_normalized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time object distance estimation")
    parser.add_argument("--midas_model_type", type=str, default="dpt_hybrid", help="MiDaS model type")
    parser.add_argument("--midas_pretrained_path", type=str, default="dpt_hybrid_384.pt", help="Path to MiDaS pretrained weights")
    parser.add_argument("--yolo_model_path", type=str, default="yolov5s.pt", help="Path to YOLO model")
    parser.add_argument("--selected_classes", type=str, default=None, help="Comma-separated list of class names to detect (e.g., 'person,car')")
    parser.add_argument("--skip_frames", type=int, default=1, help="Number of frames to skip")
    args = parser.parse_args()


    main(args.midas_model_type, args.midas_pretrained_path, args.yolo_model_path, args.selected_classes, args.skip_frames)