import os
import sys
from typing import Any

import loguru


def get_logger(module_name: str = 'Proto-FPC') -> Any:
    logger = loguru.logger.bind(task=module_name)
    try:
        loguru.logger.remove(0)
    except ValueError:
        pass

    log_format = f"| <blue>{module_name}</blue> | <green>{{time:YYYY-MM-DD-HH-mm-ss}}</green> | <level>{{level}}</level> | <cyan>{{name}}</cyan> | <level>{{file.path}}:{{line}}</level> | <level>{{message}}</level>"

    level = 'DEBUG' if os.getenv('FPC_DEBUG', '0') == '1' else 'INFO'
    level = 'TRACE' if os.getenv('FPC_TRACE', '0') == '1' else level

    def module_filter(record: Any) -> bool:
        return record['extra'].get('task') == module_name

    logger.add(
        sys.stdout,
        format=log_format,
        filter=module_filter,
        colorize=True,
        level=level,
        backtrace=True,
        diagnose=True,
    )

    return logger


logger = get_logger()
