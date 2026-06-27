import os
import glob
import importlib
from src.detectors.detector_factory import DetectorFactory

# Automatically scan all python modules in the detectors directory and import them
# to trigger the @DetectorFactory.register registration decorator.
_detectors_dir = os.path.dirname(__file__)
_module_files = glob.glob(os.path.join(_detectors_dir, "*.py"))

for _file in _module_files:
    _name = os.path.basename(_file)[:-3]
    # Skip non-detector utility files
    if _name not in ["__init__", "base_detector", "detector_factory"]:
        module_name = f"src.detectors.{_name}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            # log warning but don't fail import, since other detectors might work fine
            import logging
            logging.getLogger("detector_factory").warning(
                f"Failed to auto-load detector module {module_name}: {e}"
            )
