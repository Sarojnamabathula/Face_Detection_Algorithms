import argparse
import sys
import os
import json
import time
import numpy as np
import cv2

# Add project root to path to allow importing src package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.logger import get_logger
from src.utils import measure_resource_usage
from src.detectors.detector_factory import DetectorFactory

# Import detectors package to trigger dynamic auto-registration of plugins
import src.detectors

logger = get_logger("main")

def load_custom_config(config_path):
    """Loads a JSON config file and overrides default parameters in src/config.py."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    logger.info(f"Loading custom configuration from {config_path}...")
    try:
        with open(config_path, "r") as f:
            custom_config = json.load(f)
            
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Config override: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key ignored: {key}")
    except Exception as e:
        logger.error(f"Error parsing custom config file: {e}")
        raise

def run_single_detector(detector_name, image_path, output_dir):
    """Runs a single detector on the input image and saves outputs."""
    logger.info(f"Initializing detector: {detector_name}")
    try:
        detector = DetectorFactory.create(detector_name)
    except Exception as e:
        logger.error(f"Failed to create detector '{detector_name}': {e}")
        sys.exit(1)
        
    # Read the image
    logger.info(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image from: {image_path}")
        sys.exit(1)
        
    # Run preprocess, detection and performance tracking
    logger.info("Running face detection...")
    start_resources = measure_resource_usage()
    start_time = time.perf_counter()
    
    try:
        preprocessed = detector.preprocess_image(image)
        bboxes, confs = detector.detect_faces(preprocessed)
    except Exception as e:
        logger.error(f"Error occurred during face detection: {e}")
        sys.exit(1)
        
    end_time = time.perf_counter()
    end_resources = measure_resource_usage()
    
    elapsed_time_ms = (end_time - start_time) * 1000
    cpu_time_diff = end_resources["cpu_time"] - start_resources["cpu_time"]
    
    # Calculate CPU usage percentage relative to elapsed wall time
    elapsed_wall_s = end_resources["timestamp"] - start_resources["timestamp"]
    if elapsed_wall_s > 0:
        cpu_usage_pct = (cpu_time_diff / elapsed_wall_s) * 100
    else:
        cpu_usage_pct = 0.0
        
    memory_diff_mb = max(0.0, end_resources["rss_mb"] - start_resources["rss_mb"])
    
    # Generate output
    output_data = detector.generate_output(
        bounding_boxes=bboxes,
        confidences=confs,
        processing_time_ms=elapsed_time_ms,
        cpu_usage=cpu_usage_pct,
        memory_usage=memory_diff_mb
    )
    
    # Draw boxes
    annotated_image = detector.draw_bounding_boxes(image, bboxes, confs)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(output_dir, f"{base_name}_{detector_name}_output.json")
    img_path = os.path.join(output_dir, f"{base_name}_{detector_name}_result.jpg")
    
    # Save JSON file
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Saved JSON output to: {json_path}")
    
    # Save annotated image
    cv2.imwrite(img_path, annotated_image)
    logger.info(f"Saved annotated image to: {img_path}")
    
    # Print JSON output to stdout as requested
    print(json.dumps(output_data, indent=4))
    return output_data

def run_comparison(image_path, output_dir):
    """Runs all registered detectors on the same image and prints/saves comparative stats."""
    logger.info("Initializing comparison mode...")
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image from: {image_path}")
        sys.exit(1)
        
    available_detectors = list(DetectorFactory._detectors.keys())
    if not available_detectors:
        logger.error("No registered detectors found!")
        sys.exit(1)
        
    results = {}
    annotated_images = {}
    
    for name in available_detectors:
        logger.info(f"--- Running detector: {name.upper()} ---")
        try:
            detector = DetectorFactory.create(name)
            
            start_resources = measure_resource_usage()
            start_time = time.perf_counter()
            
            preprocessed = detector.preprocess_image(image)
            bboxes, confs = detector.detect_faces(preprocessed)
            
            end_time = time.perf_counter()
            end_resources = measure_resource_usage()
            
            elapsed_time_ms = (end_time - start_time) * 1000
            cpu_time_diff = end_resources["cpu_time"] - start_resources["cpu_time"]
            elapsed_wall_s = end_resources["timestamp"] - start_resources["timestamp"]
            cpu_usage_pct = (cpu_time_diff / elapsed_wall_s) * 100 if elapsed_wall_s > 0 else 0.0
            memory_diff_mb = max(0.0, end_resources["rss_mb"] - start_resources["rss_mb"])
            
            output_data = detector.generate_output(
                bounding_boxes=bboxes,
                confidences=confs,
                processing_time_ms=elapsed_time_ms,
                cpu_usage=cpu_usage_pct,
                memory_usage=memory_diff_mb
            )
            
            # Inject metadata properties for comparison display
            output_data["accuracy_rating"] = detector.accuracy_rating
            output_data["cpu_usage_rating"] = detector.cpu_usage_rating
            
            results[name] = output_data
            
            # Render visual output
            annotated = detector.draw_bounding_boxes(image, bboxes, confs)
            annotated_images[name] = annotated
            
            # Save individual detector results in comparison run
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_{name}_result.jpg"), annotated)
            
        except Exception as e:
            logger.error(f"Detector '{name}' failed during comparison: {e}")
            
    if not results:
        logger.error("All detectors failed to run in comparison mode.")
        sys.exit(1)
        
    # Generate a side-by-side visualization if we have Haar and MTCNN (or any two detectors)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    if len(annotated_images) >= 2:
        # Get the first two detectors for side-by-side concatenation
        detector_keys = list(annotated_images.keys())[:2]
        img1 = annotated_images[detector_keys[0]]
        img2 = annotated_images[detector_keys[1]]
        
        # Check shapes, resize if mismatch (they should match since it's the same source image)
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
        combined_img = np.hstack((img1, img2))
        combined_path = os.path.join(output_dir, f"{base_name}_comparison_side_by_side.jpg")
        cv2.imwrite(combined_path, combined_img)
        logger.info(f"Saved side-by-side comparison image to: {combined_path}")
        
    # Print the comparison summary table in Markdown format
    print("\n=================== COMPARISON SUMMARY ===================")
    print("\n| Metric             | " + " | ".join([f"{name.upper()}" for name in results.keys()]) + " |")
    print("| ------------------ |" + "|".join([" :---: " for _ in results]) + "|")
    
    # 1. Faces Detected
    row_faces = "| Faces Detected     |"
    for name, data in results.items():
        row_faces += f" {data['faces_detected']} |"
    print(row_faces)
    
    # 2. Average Confidence
    row_conf = "| Average Confidence |"
    for name, data in results.items():
        confs = data["confidence"]
        avg_c = sum(confs) / len(confs) if confs else 0.0
        row_conf += f" {avg_c:.2f} |"
    print(row_conf)
    
    # 3. Processing Time
    row_time = "| Processing Time    |"
    for name, data in results.items():
        t = data["metrics"]["processing_time_ms"]
        row_time += f" {t:.1f} ms |"
    print(row_time)
    
    # 4. CPU Usage
    row_cpu = "| CPU Usage          |"
    for name, data in results.items():
        row_cpu += f" {data['cpu_usage_rating']} |"
    print(row_cpu)
    
    # 5. Accuracy
    row_acc = "| Accuracy           |"
    for name, data in results.items():
        row_acc += f" {data['accuracy_rating']} |"
    print(row_acc)
    print("\n===========================================================")
    
    # Save comparison data JSON
    comp_json_path = os.path.join(output_dir, f"{base_name}_comparison_metrics.json")
    with open(comp_json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved comparison JSON metrics to: {comp_json_path}\n")

def run_camera_comparison(detector_name="compare", camera_src=0):
    """Captures video from the webcam/droidcam and runs face detection in real-time."""
    logger.info(f"Initializing camera feed from source: {camera_src}...")
    cap = cv2.VideoCapture(camera_src)
    if not cap.isOpened():
        logger.error(f"Could not open camera source: {camera_src}")
        return
        
    logger.info("Camera opened successfully. Press 'q' in the display window to exit.")
    
    # Instantiate detectors
    detectors = {}
    if detector_name == "compare":
        # Get all registered detectors
        for name in DetectorFactory._detectors.keys():
            try:
                detector = DetectorFactory.create(name)
                detectors[name] = detector
            except Exception as e:
                logger.warning(f"Could not load detector {name} for live camera: {e}")
    else:
        try:
            detector = DetectorFactory.create(detector_name)
            detectors[detector_name] = detector
        except Exception as e:
            logger.error(f"Could not create detector {detector_name}: {e}")
            cap.release()
            return
            
    if not detectors:
        logger.error("No active detectors available.")
        cap.release()
        return

    logger.info(f"Starting live stream loop using: {list(detectors.keys())}")
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame.")
            break
            
        annotated_frames = {}
        
        # Process frame for each detector
        for name, detector in detectors.items():
            try:
                preprocessed = detector.preprocess_image(frame)
                
                start_t = time.perf_counter()
                bboxes, confs = detector.detect_faces(preprocessed)
                end_t = time.perf_counter()
                
                # Annotate frame
                annotated = detector.draw_bounding_boxes(frame, bboxes, confs)
                
                # Add FPS label
                fps = 1.0 / (end_t - start_t) if (end_t - start_t) > 0 else 0.0
                cv2.putText(
                    annotated, 
                    f"{name.upper()} - FPS: {fps:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0) if name == "haar" else (0, 0, 255), 
                    2
                )
                annotated_frames[name] = annotated
            except Exception as e:
                # Fallback to original frame on error
                annotated_frames[name] = frame.copy()
                
        # Combine frames side-by-side if doing comparison
        if len(annotated_frames) >= 2:
            keys = list(annotated_frames.keys())[:2]
            img1 = annotated_frames[keys[0]]
            img2 = annotated_frames[keys[1]]
            
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                
            combined = np.hstack((img1, img2))
        elif len(annotated_frames) == 1:
            combined = list(annotated_frames.values())[0]
        else:
            combined = frame
            
        try:
            cv2.imshow("Live Face Detection - Press 'q' to Quit", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            logger.error(
                "GUI window could not be opened. This occurs when running in a headless "
                "environment or over SSH. Please run the script directly on the host computer's terminal. "
                f"Error: {e}"
            )
            break
            
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam feed closed.")

def main():
    parser = argparse.ArgumentParser(description="Multi-Algorithm Face Detection CLI")
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to the input image for face detection"
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use webcam/droidcam for live face detection comparison"
    )
    parser.add_argument(
        "--camera-src",
        type=str,
        default="0",
        help="Camera device index (e.g. 0, 1, 2) or network stream URL (for DroidCam MJPEG feed)"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        help="Detection algorithm: 'haar', 'mtcnn', or 'compare' to run all."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to a custom JSON configuration file overriding defaults."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=config.OUTPUT_DIR, 
        help="Directory to save the processed outputs"
    )
    
    args = parser.parse_args()
    
    # Check that either image or camera is provided
    if not args.image and not args.camera:
        parser.error("one of the arguments --image or --camera is required")
        
    # Load custom configuration if provided
    if args.config:
        try:
            load_custom_config(args.config)
        except Exception as e:
            sys.exit(1)
            
    # Resolve selected algorithm (command-line takes precedence over config)
    selected_algo = args.algorithm if args.algorithm else config.DETECTION_ALGORITHM
    selected_algo = selected_algo.lower()
    
    if args.camera:
        # Convert index to integer if numeric
        cam_src = args.camera_src
        if cam_src.isdigit():
            cam_src = int(cam_src)
        run_camera_comparison(selected_algo, cam_src)
    else:
        if selected_algo == "compare":
            run_comparison(args.image, args.output_dir)
        else:
            run_single_detector(selected_algo, args.image, args.output_dir)

if __name__ == "__main__":
    main()
