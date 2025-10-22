import logging


def setup_logger(name=None):
    """Sets up the logger for the entire run."""
    # Suppress verbose logs from py4j
    logging.getLogger("py4j").setLevel(logging.ERROR)
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

    # Create or get a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the default log level
    logger.propagate = False  # Disable logger propagation to prevent duplicate logs
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    MSG_FORMAT = "%(asctime)s %(levelname)-8s [%(filename)s:%(funcName)s():%(lineno)d] %(message)s"

    # Check if the logger already has handlers to avoid duplicate logs
    if not logger.hasHandlers():
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(MSG_FORMAT, DATE_FORMAT)
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    return logger
