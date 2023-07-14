import logging
import time


def consecutive_timing():
    basetime = time.time()

    def cp_time(prefix: str):
        nonlocal basetime
        now = time.time()
        logging.debug(f'{prefix}: {now - basetime}')
        basetime = now

    return cp_time


CHECK_TIME = consecutive_timing()
