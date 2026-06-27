import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure project root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy modules before importing code that imports them
mock_cv2 = MagicMock()
mock_mtcnn = MagicMock()
mock_tf = MagicMock()

# Setup CascadeClassifier mock behavior
mock_cascade = MagicMock()
mock_cascade.empty.return_value = False
# detectMultiScale3 returns (rectangles, rejectLevels, levelWeights)
mock_cascade.detectMultiScale3.return_value = (
    [[10, 20, 30, 40], [50, 60, 70, 80]],
    [1.0, 2.0],
    [[10.0], [20.0]]
)
mock_cv2.CascadeClassifier.return_value = mock_cascade
mock_cv2.cvtColor.return_value = MagicMock() # Mock image conversion

# Setup MTCNN mock behavior
mock_mtcnn_instance = MagicMock()
mock_mtcnn_instance.detect_faces.return_value = [
    {"box": [15, 25, 35, 45], "confidence": 0.98},
    {"box": [55, 65, 75, 85], "confidence": 0.95}
]
mock_mtcnn.MTCNN.return_value = mock_mtcnn_instance

sys.modules["cv2"] = mock_cv2
sys.modules["mtcnn"] = mock_mtcnn
sys.modules["tensorflow"] = mock_tf

# Now we can import the code we want to test
from src.detectors.detector_factory import DetectorFactory
from src.detectors.haar_detector import HaarDetector
from src.detectors.mtcnn_detector import MTCNNDetector
from src.detectors.base_detector import BaseDetector
from src import config

class TestDetectorFramework(unittest.TestCase):
    
    def setUp(self):
        # Reset mock call counts
        mock_cascade.reset_mock()
        mock_mtcnn_instance.reset_mock()
        mock_cv2.reset_mock()
        
    def test_factory_registration(self):
        """Test that detectors are successfully registered and retrieved from factory."""
        self.assertIn("haar", DetectorFactory._detectors)
        self.assertIn("mtcnn", DetectorFactory._detectors)
        
        haar_det = DetectorFactory.create("haar")
        self.assertIsInstance(haar_det, HaarDetector)
        self.assertEqual(haar_det.algorithm_name, "haar")
        self.assertEqual(haar_det.accuracy_rating, "Moderate")
        self.assertEqual(haar_det.cpu_usage_rating, "Low")
        
        mtcnn_det = DetectorFactory.create("mtcnn")
        self.assertIsInstance(mtcnn_det, MTCNNDetector)
        self.assertEqual(mtcnn_det.algorithm_name, "mtcnn")
        self.assertEqual(mtcnn_det.accuracy_rating, "High")
        self.assertEqual(mtcnn_det.cpu_usage_rating, "Medium")

    def test_factory_invalid_detector(self):
        """Test factory handles invalid detector names by raising ValueError."""
        with self.assertRaises(ValueError):
            DetectorFactory.create("invalid_name")

    def test_haar_detector_preprocess_and_detect(self):
        """Test preprocess, detect, and output generation of Haar Cascade detector."""
        detector = DetectorFactory.create("haar")
        
        # Test image path loading (we mock cv2.imread inside the test)
        dummy_img = MagicMock()
        mock_cv2.imread.return_value = dummy_img
        
        preprocessed = detector.preprocess_image("dummy_path.jpg")
        mock_cv2.imread.assert_called_with("dummy_path.jpg")
        self.assertIn("gray", preprocessed)
        
        # Test detection
        boxes, confs = detector.detect_faces(preprocessed)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], [10, 20, 30, 40])
        self.assertEqual(boxes[1], [50, 60, 70, 80])
        self.assertEqual(len(confs), 2)
        
        # Test JSON output generation
        out = detector.generate_output(boxes, confs, 15.5, cpu_usage=5.0, memory_usage=1.2)
        self.assertEqual(out["algorithm"], "haar")
        self.assertEqual(out["faces_detected"], 2)
        self.assertEqual(out["bounding_boxes"], [[10, 20, 30, 40], [50, 60, 70, 80]])
        self.assertEqual(out["confidence"], confs)
        self.assertEqual(out["metrics"]["processing_time_ms"], 15.5)
        self.assertEqual(out["metrics"]["cpu_usage_pct"], 5.0)
        self.assertEqual(out["metrics"]["memory_usage_mb"], 1.2)

    def test_mtcnn_detector_preprocess_and_detect(self):
        """Test preprocess, detect, and output generation of MTCNN detector."""
        detector = DetectorFactory.create("mtcnn")
        
        # Test image numpy array preprocessing
        dummy_img = MagicMock()
        preprocessed = detector.preprocess_image(dummy_img)
        self.assertIn("rgb", preprocessed)
        
        # Test detection
        boxes, confs = detector.detect_faces(preprocessed)
        mock_mtcnn_instance.detect_faces.assert_called_once()
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0], [15, 25, 35, 45])
        self.assertEqual(boxes[1], [55, 65, 75, 85])
        self.assertEqual(confs, [0.98, 0.95])
        
        # Test JSON output generation
        out = detector.generate_output(boxes, confs, 45.2, cpu_usage=25.0, memory_usage=55.4)
        self.assertEqual(out["algorithm"], "mtcnn")
        self.assertEqual(out["faces_detected"], 2)
        self.assertEqual(out["bounding_boxes"], [[15, 25, 35, 45], [55, 65, 75, 85]])
        self.assertEqual(out["confidence"], [0.98, 0.95])
        self.assertEqual(out["metrics"]["processing_time_ms"], 45.2)

    def test_detector_drawing_boxes(self):
        """Test draw_bounding_boxes outputs an annotated image."""
        haar = DetectorFactory.create("haar")
        dummy_img = MagicMock()
        dummy_img.copy.return_value = dummy_img
        
        # Mock rectangle and text draw functions
        mock_cv2.rectangle = MagicMock()
        mock_cv2.putText = MagicMock()
        
        result_img = haar.draw_bounding_boxes(dummy_img, [[10, 20, 30, 40]], [0.9])
        
        mock_cv2.rectangle.assert_called_once()
        mock_cv2.putText.assert_called_once()
        self.assertEqual(result_img, dummy_img)

if __name__ == "__main__":
    unittest.main()
