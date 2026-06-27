from src.detectors.base_detector import BaseDetector

class DetectorFactory:
    """Factory class to create face detectors dynamically based on the selected algorithm name."""
    
    _detectors = {}
    
    @classmethod
    def register(cls, name):
        """Decorator to register a detector class under a unique identifier name.
        
        Args:
            name (str): The unique name of the detector (e.g., 'haar', 'mtcnn').
            
        Returns:
            decorator function.
        """
        def decorator(detector_class):
            cls._detectors[name.lower()] = detector_class
            return detector_class
        return decorator
        
    @classmethod
    def create(cls, name) -> BaseDetector:
        """Instantiates and returns the detector registered under the given name.
        
        Args:
            name (str): The name of the detector to create.
            
        Returns:
            BaseDetector: The instantiated detector.
            
        Raises:
            ValueError: If the detector name is not registered.
        """
        name_lower = name.lower()
        if name_lower not in cls._detectors:
            # Try lazy auto-loading by importing detectors package if not already populated
            import src.detectors
            
        if name_lower not in cls._detectors:
            raise ValueError(
                f"Unknown detector algorithm '{name}'. "
                f"Supported detectors are: {list(cls._detectors.keys())}"
            )
        
        detector_class = cls._detectors[name_lower]
        return detector_class()
