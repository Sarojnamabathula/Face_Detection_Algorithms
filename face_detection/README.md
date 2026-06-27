# Multi-Algorithm Face Detection Framework

This is a modular and configurable face detection system supporting multiple algorithms. It allows you to select, configure, and compare face detectors seamlessly through a unified interface.

The design implements the **Factory Pattern** and uses a **dynamic plugin auto-loader**, allowing new face detectors (e.g. YuNet, YOLO Face, RetinaFace, MediaPipe) to be added without modifying the core system or CLI interface.

---

## Folder Structure

```text
face_detection/
├── models/
│   ├── haarcascade_frontalface_default.xml
│   └── mtcnn/
│
├── src/
│   ├── detectors/
│   │   ├── base_detector.py
│   │   ├── haar_detector.py
│   │   ├── mtcnn_detector.py
│   │   └── detector_factory.py
│   │
│   ├── utils.py
│   ├── config.py
│   └── logger.py
│
├── images/
├── output/
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

1. Clone or copy this directory to your machine.
2. Install the python dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: MTCNN requires TensorFlow. Installing `tensorflow-cpu` is recommended if running on resource-constrained systems or systems without a dedicated GPU.*

---

## Usage

You can run the face detection pipeline from the command line using `main.py`.

### 1. Run Haar Cascade (Default)
By default, the script loads the detector configured in `config.py`. To explicitly use the OpenCV Haar Cascade classifier:
```bash
python main.py --image images/my_image.jpg --algorithm haar
```
*Note: If the Haar Cascade XML classifier template file is not present in the `models/` directory, it will automatically download from OpenCV's repository on the first run.*

### 2. Run MTCNN Detector
To run the deep learning MTCNN detector (which outputs faces and confidences):
```bash
python main.py --image images/my_image.jpg --algorithm mtcnn
```

### 3. Run Comparison Mode
To execute all registered algorithms on the same image, write output visualizations to the `output/` directory, and print a formatted markdown comparison table to stdout:
```bash
python main.py --image images/my_image.jpg --algorithm compare
```

### 4. Custom Configuration File
You can override parameters like scale factor, thresholds, or default algorithm selection by creating a custom JSON configuration file and passing it with `--config`:
```bash
python main.py --image images/my_image.jpg --config custom_config.json
```
Example `custom_config.json`:
```json
{
    "DETECTION_ALGORITHM": "mtcnn",
    "HAAR_SCALE_FACTOR": 1.2,
    "HAAR_MIN_NEIGHBORS": 4
}
```

---

## Extensibility: Adding a New Detector

Adding a new detector is extremely easy and requires **no modifications** to `main.py` or the `DetectorFactory` code.

### Step 1: Create a new python file
Create a file inside `src/detectors/` (e.g. `src/detectors/yolo_detector.py`).

### Step 2: Implement the BaseDetector class and register it
Import `BaseDetector` and register your class using the `@DetectorFactory.register("algorithm_name")` decorator.

Example:
```python
import cv2
from src.detectors.base_detector import BaseDetector
from src.detectors.detector_factory import DetectorFactory

@DetectorFactory.register("yolo")
class YoloFaceDetector(BaseDetector):
    @property
    def algorithm_name(self):
        return "yolo"
        
    @property
    def accuracy_rating(self):
        return "High"
        
    @property
    def cpu_usage_rating(self):
        return "High"
        
    def load_model(self):
        # Load weights, configuration files, etc.
        pass
        
    def preprocess_image(self, image_path_or_array):
        # Convert BGR image array if needed
        pass
        
    def detect_faces(self, preprocessed_image):
        # Return boxes in format [x, y, w, h] and confidences in format [0.0 - 1.0]
        bounding_boxes = [[100, 120, 50, 50]]
        confidences = [0.95]
        return bounding_boxes, confidences
        
    def draw_bounding_boxes(self, original_image, bounding_boxes, confidences):
        # Draw boxes on image
        pass
        
    def generate_output(self, bounding_boxes, confidences, processing_time_ms, cpu_usage=None, memory_usage=None):
        # Generate standardized JSON output dict
        return {
            "algorithm": self.algorithm_name,
            "faces_detected": len(bounding_boxes),
            "bounding_boxes": bounding_boxes,
            "confidence": confidences,
            "metrics": {
                "processing_time_ms": processing_time_ms
            }
        }
```

Because of the dynamic registration in `src/detectors/__init__.py`, this new detector is automatically recognized. You can run it immediately:
```bash
python main.py --image images/my_image.jpg --algorithm yolo
```
And it will also be dynamically included in the **Comparison Mode** reports!

---

## Unit Testing
We provide mock-based unit tests to verify the system without needing heavy external dependencies:
```bash
python -m unittest face_detection.tests.test_detectors
```
