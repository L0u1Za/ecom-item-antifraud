import logging

# Configure the logger
def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger('fraud_detection', 'fraud_detection.log')
    logger.info('Logger is set up and ready to use.')