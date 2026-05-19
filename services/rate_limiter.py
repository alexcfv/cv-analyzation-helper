import threading
import time


class RateLimiter:
    def __init__(self, min_interval: float = 1.2):
        self.min_interval = min_interval
        self.last_call_time: float = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()
