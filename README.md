# DepthSense-Fusion

This project provides real-time distance and depth estimation capabilities by combining the **MiDaS** depth estimation model with **YOLOv5** object detection. It helps compute the distance to objects in an image or live video feed and overlays the depth map alongside object bounding boxes.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-MiDaS-orange)](https://github.com/isl-org/MiDaS)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-00FFFF?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/yolov5)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-0142D3.svg?logo=intel&logoColor=white)](https://docs.openvino.ai/)

## Key Features
- **Depth Estimation**: Powered by MiDaS, creating high-quality dense depth maps.
- **Object Detection**: Leverages YOLOv5 to detect objects in real-time.
- **Distance Calculation**: Combines object bounds with depth maps to estimate the approximate distance to detected objects.
- **Versatile Inference Options**: Supports multiple modes of running inference, from basic PyTorch execution to optimized OpenVINO processing.

---

## Core scripts

- `run_basic_inference.py`
  - A foundational script that runs MiDaS (`model-f6b98070.pt`) and YOLOv5 (`yolov5s.pt`) simply and directly. It calculates and draws bounding boxes with the rough depth values on a live webcam feed.

- `run_openvino.py`
  - Designed for CPU-accelerated performance using Intel's OpenVINO toolkit. It loads a pre-compiled OpenVINO MiDaS model (`openvino_midas_v21_small.xml`) and runs purely on the CPU, making it ideal for edge devices or non-GPU systems.
---

## OpenVINO vs. PyTorch

- **PyTorch (`run_dpt_optimized.py`, `run_basic_inference.py`)**: 
  These scripts run native PyTorch models (`.pt` files). They are best when you have an NVIDIA GPU available, as PyTorch can leverage CUDA for massive parallel speedups.
- **OpenVINO (`run_openvino.py`)**: 
  OpenVINO runs an optimized Intermediate Representation (IR) model (`.xml` / `.bin`). It is specifically designed to drastically accelerate deep learning on **Intel CPUs** and integrated GPUs. Choose this script if you are deploying on a CPU-only machine or an edge device where PyTorch is too slow.

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/Mohamed-dev1/DeptMaps-MiDaS.git
cd DeptMaps-MiDaS
```

**2. Install dependencies**
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

**3. Model Files (Important)**
The following files are **not included** in this repository:
- `src/models/dpt_hybrid_384.pt`
- `src/models/model-f6b98070.pt`

**Where to get them:**
- **MiDaS v2.1 Small (`model-f6b98070.pt`)**: Download from the official [MiDaS Releases](https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.pt).
- **MiDaS DPT Hybrid (`dpt_hybrid_384.pt`)**: Download from the [MiDaS Releases](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt).

Place these `.pt` files inside the `src/models/` directory before running the scripts.

---

## Usage

You can run any of the scripts depending on your hardware and objective. By default, they will try to open your webcam (device `0`).

**Run the basic inference:**
```bash
python run_basic_inference.py
```

**Run the high-quality DPT model (Requires downloaded weights):**
```bash
python run_dpt_optimized.py
```

**Run the OpenVINO optimized version:**
```bash
python run_openvino.py
```

Press `q` within the live feed window to quit any of the scripts.

---

## Additional Links
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [MiDaS GitHub](https://github.com/isl-org/MiDaS)
- [OpenVINO Documentation](https://docs.openvino.ai/)#
