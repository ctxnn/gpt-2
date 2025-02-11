import sys
import datetime
from contextlib import contextmanager

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{filename}_{self.timestamp}.txt"
        self.log = open(self.filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

@contextmanager
def capture_training_output(filename="training_log"):
    logger = Logger(filename)
    _stdout = sys.stdout
    sys.stdout = logger
    try:
        yield logger
    finally:
        sys.stdout = _stdout
        logger.close()

sys.stdout = Logger("training_output")
