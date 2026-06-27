import cv2
import numpy as np
import os
from src.detectors.base_detector import BaseDetector
from src.detectors.detector_factory import DetectorFactory
from src import config
from src.utils import download_file
from src.logger import get_logger

logger = get_logger("haar_detector")

@DetectorFactory.register("haar")
class HaarDetector(BaseDetector):
    """Haar Cascade face detector based on OpenCV CascadeClassifier."""
    
    @property
    def algorithm_name(self):
        return "haar"
        
    @property
    def accuracy_rating(self):
        return "Moderate"
        
    @property
    def cpu_usage_rating(self):
        return "Low"
        
    def load_model(self):
        # Automatically download XML if it doesn't exist
        if not os.path.exists(config.HAAR_CASCADE_PATH):
            logger.info("Haar Cascade XML template not found. Downloading...")
            download_file(config.HAAR_CASCADE_URL, config.HAAR_CASCADE_PATH)
            
        self.model = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
        if self.model.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {config.HAAR_CASCADE_PATH}")
        logger.info("Haar Cascade model loaded successfully.")
        
    def preprocess_image(self, image_path_or_array):
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise FileNotFoundError(f"Image not found at path: {image_path_or_array}")
        else:
            image = image_path_or_array.copy()
            
        # Convert BGR (OpenCV default) to Grayscale for Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return {"original": image, "gray": gray}
        
    def detect_faces(self, preprocessed_image):
        gray = preprocessed_image["gray"]
        
        bounding_boxes = []
        confidences = []
        
        try:
            # Using detectMultiScale3 to retrieve rejectLevels and levelWeights (for confidence)
            rects, rejectLevels, levelWeights = self.model.detectMultiScale3(
                gray,
                scaleFactor=config.HAAR_SCALE_FACTOR,
                minNeighbors=config.HAAR_MIN_NEIGHBORS,
                minSize=config.HAAR_MIN_SIZE,
                outputRejectLevels=True
            )
            
            if len(rects) > 0:
                # Flat list representation of weights
                weights_flat = [float(w[0]) if isinstance(w, (list, np.ndarray)) else float(w) for w in levelWeights]
                max_w = max(weights_flat) if weights_flat else 1.0
                min_w = min(weights_flat) if weights_flat else 0.0
                w_range = max_w - min_w
                
                for rect, weight in zip(rects, weights_flat):
                    x, y, w, h = rect
                    bounding_boxes.append([int(x), int(y), int(w), int(h)])
                    
                    # Normalize weight to a mock confidence value in [0.5, 0.99]
                    if w_range > 0:
                        conf = 0.5 + 0.49 * ((weight - min_w) / w_range)
                    else:
                        conf = 0.90
                    confidences.append(float(round(conf, 2)))
        except Exception as e:
            logger.debug(f"detectMultiScale3 is unavailable or failed, falling back to detectMultiScale. Reason: {e}")
            faces = self.model.detectMultiScale(
                gray,
                scaleFactor=config.HAAR_SCALE_FACTOR,
                minNeighbors=config.HAAR_MIN_NEIGHBORS,
                minSize=config.HAAR_MIN_SIZE
            )
            for (x, y, w, h) in faces:
                bounding_boxes.append([int(x), int(y), int(w), int(h)])
                confidences.append(0.90)
                
        return bounding_boxes, confidences
        
    def draw_bounding_boxes(self, original_image, bounding_boxes, confidences):
        img_copy = original_image.copy()
        for box, conf in zip(bounding_boxes, confidences):
            x, y, w, h = box
            # Draw green bounding box (traditional BGR: 0, 255, 0)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Label
            label = f"Haar: {conf:.2f}"
            cv2.putText(img_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
