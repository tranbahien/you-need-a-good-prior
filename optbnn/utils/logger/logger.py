"""Utilites functions for the log files."""

import os
import logging
import logging.config

from pathlib import Path

from ..util import read_json


def setup_logging(save_dir, name='logger', verbosity=2,
                  log_config='./logger/logger_config.json',
                  default_level=logging.INFO):
    """Setup logging configuration and get logger
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = os.path.join(save_dir,
                                                   handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(
            log_config))
        logging.basicConfig(level=default_level)

    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG
    }

    if not (verbosity in log_levels.keys()):
        level = default_level
    else:
        level = log_levels[verbosity]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
