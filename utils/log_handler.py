import logging
import os
import sys
import time

def get_logger(log_dir, log_filename='info.log', name="experiment", level=logging.INFO):
    """
    Create a logger that writes to both the console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers.
    if logger.handlers:
        return logger
    
    # Formatter.
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Use the provided filename for reproducible experiment logs.
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    
    # Console handler.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
