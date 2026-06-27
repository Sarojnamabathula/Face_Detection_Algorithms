import cv2
import numpy as np
from src.detectors.base_detector import BaseDetector
from src.detectors.detector_factory import DetectorFactory
from src import config
from src.logger import get_logger

logger = get_logger("mtcnn_detector")

@DetectorFactory.register("mtcnn")
class MTCNNDetector(BaseDetector):
    """MTCNN deep learning-based face detector."""
    
    @property
    def algorithm_name(self):
        return "mtcnn"
        
    @property
    def accuracy_rating(self):
        return "High"
        
    @property
    def cpu_usage_rating(self):
        return "Medium"
        
    def load_model(self):
        try:
            # Lazy import to avoid loading heavy framework dependencies on startup
            from mtcnn import MTCNN
            try:
                self.model = MTCNN(
                    min_face_size=config.MTCNN_MIN_FACE_SIZE,
                    steps_threshold=config.MTCNN_STEPS_THRESHOLD,
                    scale_factor=config.MTCNN_SCALE_FACTOR
                )
            except TypeError:
                logger.info("MTCNN constructor does not accept parameters. Falling back to default instantiation.")
                self.model = MTCNN()
            logger.info("MTCNN model loaded successfully.")
        except ImportError as e:
            logger.error(
                "Failed to import 'mtcnn' or 'tensorflow'. "
                "MTCNN requires TensorFlow and MTCNN packages. "
                "Please run: pip install tensorflow mtcnn"
            )
            raise RuntimeError(
                "MTCNN dependencies (tensorflow, mtcnn) are not installed in the environment."
            ) from e
            
    def preprocess_image(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise FileNotFoundError(f"Image not found at path: {image_path_or_array}")
        else:
            image = image_path_or_array.copy()
            
        # MTCNN expects RGB images, OpenCV standard is BGR
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return {"original": image, "rgb": rgb}
        
    def detect_faces(self, preprocessed_image):
        rgb = preprocessed_image["rgb"]
        
        # MTCNN returns a list of dictionaries, e.g.:
        # [{'box': [x, y, w, h], 'confidence': 0.998, 'keypoints': {...}}, ...]
        results = self.model.detect_faces(rgb)
        
        bounding_boxes = []
        confidences = []
        
        for res in results:
            box = res["box"]  # [x, y, width, height]
            conf = res["confidence"]
            bounding_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            confidences.append(float(round(conf, 2)))
            
        return bounding_boxes, confidences
        
    def draw_bounding_boxes(self, original_image, bounding_boxes, confidences):
        img_copy = original_image.copy()
        for box, conf in zip(bounding_boxes, confidences):
            x, y, w, h = box
            # Draw red bounding box (deep learning default: BGR: 0, 0, 255)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Label
            label = f"MTCNN: {conf:.2f}"
            cv2.putText(img_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return img_copy
        
    def generate_output(self, bounding_boxes, confidences, processing_time_ms, cpu_usage=None, memory_usage=None):
        out = {
            "algorithm": self.algorithm_name,
            "faces_detected": len(bounding_boxes),
            "bounding_boxes": bounding_boxes,
            "confidence": confidences
        }
        # Add metrics if available
        if processing_time_ms is not None:
            out["metrics"] = {
                "processing_time_ms": round(processing_time_ms, 2)
            }
            if cpu_usage is not None:
                out["metrics"]["cpu_usage_pct"] = round(cpu_usage, 2) if isinstance(cpu_usage, (int, float)) else cpu_usage
            if memory_usage is not None:
                out["metrics"]["memory_usage_mb"] = round(memory_usage, 2) if isinstance(memory_usage, (int, float)) else memory_usage
        return out
