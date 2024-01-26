import logging


def pytest_configure(config):
    # Logging stuff
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)

    # Custom markers
    config.addinivalue_line("markers", "dataset: mark tests that assess the integrity of our datasets.")
    config.addinivalue_line("markers", "mnist: mark tests that are related to our mnist example")
    config.addinivalue_line("markers", "gpt: mark tests that are related to our gpt example")
