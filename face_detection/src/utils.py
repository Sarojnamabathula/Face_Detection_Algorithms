import os
import urllib.request
import time
from src.logger import get_logger

logger = get_logger("utils")

try:
    import psutil
except ImportError:
    psutil = None
    logger.warning("psutil is not installed. CPU and memory metrics will not be measured accurately.")

def download_file(url, dest_path):
    """Downloads a file from a URL to a destination path if it doesn't exist."""
    if os.path.exists(dest_path):
        return
    
    logger.info(f"Downloading file from {url} to {dest_path}...")
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # Add a user-agent to avoid HTTP 403 Forbidden
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {e}")
        raise

def measure_resource_usage():
    """Returns a dict containing current process memory (MB) and CPU times."""
    if psutil is None:
        return {
            "rss_mb": 0.0,
            "cpu_time": 0.0,
            "timestamp": time.perf_counter()
        }
    
    try:
        process = psutil.Process()
        # Memory in MB
        mem_info = process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        # CPU times (user + system)
        cpu_times = process.cpu_times()
        user_system_cpu = cpu_times.user + cpu_times.system
        return {
            "rss_mb": rss_mb,
            "cpu_time": user_system_cpu,
            "timestamp": time.perf_counter()
        }
    except Exception as e:
        logger.warning(f"Failed to measure resource usage: {e}")
        return {
            "rss_mb": 0.0,
            "cpu_time": 0.0,
            "timestamp": time.perf_counter()
        }
