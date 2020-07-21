import logging


# NO IDEA WHY CAN'T WRITE OUT DEBUG/INFO MESSAGES W/CUSTOM LOGGER HERE WITHOUT CALL BASICCONFIG()...

def get_logger(log_file) -> logging.Logger:

    "Initialize and return logger that writes to given file"


    logger = logging.getLogger(name=__name__)
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # create formatters for handlers
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    console_format = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # logger.propagate = False
    return logger



