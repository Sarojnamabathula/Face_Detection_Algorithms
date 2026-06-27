from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """Abstract Base Detector class that all face detection algorithms must implement."""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    @abstractmethod
    def load_model(self):
        """Loads the model weights, cascades, or graph dependencies."""
        pass
        
    @abstractmethod
    def preprocess_image(self, image_path_or_array):
        """Preprocesses the input image for the detector.
        
        Args:
            image_path_or_array: String path to the image, or a numpy array (BGR).
            
        Returns:
            Any preprocessed representation (usually a dict or numpy array)
            required by detect_faces.
        """
        pass
        
    @abstractmethod
    def detect_faces(self, preprocessed_image):
        """Performs face detection on the preprocessed image.
        
        Args:
            preprocessed_image: The output of preprocess_image.
            
        Returns:
            tuple: (bounding_boxes, confidences)
                - bounding_boxes: list of lists [x, y, w, h]
                - confidences: list of floats (0.0 to 1.0)
        """
        pass
        
    @abstractmethod
    def draw_bounding_boxes(self, original_image, bounding_boxes, confidences):
        """Draws bounding boxes and confidences onto the image.
        
        Args:
            original_image: numpy array of the original image (BGR).
            bounding_boxes: list of [x, y, w, h] coordinates.
            confidences: list of confidence scores.
            
        Returns:
            numpy array: Image with bounding boxes drawn.
        """
        pass
        
    @abstractmethod
    def generate_output(self, bounding_boxes, confidences, processing_time_ms, cpu_usage=None, memory_usage=None):
        """Generates standardized output format.
        
        Args:
            bounding_boxes: list of [x, y, w, h] coordinates.
            confidences: list of confidence scores.
            processing_time_ms: float, processing time in milliseconds.
            cpu_usage: optional float or string, CPU usage metric.
            memory_usage: optional float or string, Memory usage metric (MB).
            
        Returns:
            dict: Standardized output structure.
        """
        pass
    
    @property
    @abstractmethod
    def algorithm_name(self):
        """Returns the identifier name of the algorithm."""
        pass

    @property
    def accuracy_rating(self):
        """Returns the qualitative accuracy rating of the detector."""
        return "Moderate"

    @property
    def cpu_usage_rating(self):
        """Returns the qualitative CPU usage rating of the detector."""
        return "Low"
