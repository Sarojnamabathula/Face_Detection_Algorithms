import os

# Project structure paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Haar Cascade configuration
HAAR_CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
HAAR_CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# MTCNN configuration
MTCNN_MODEL_DIR = os.path.join(MODELS_DIR, "mtcnn")

# Default detection algorithm: 'haar' or 'mtcnn'
DETECTION_ALGORITHM = "haar"

# Default detector parameters
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (30, 30)

MTCNN_MIN_FACE_SIZE = 20
MTCNN_STEPS_THRESHOLD = [0.6, 0.7, 0.7]
MTCNN_SCALE_FACTOR = 0.709
