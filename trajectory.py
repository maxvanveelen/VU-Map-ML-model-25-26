import time

class Trajectory:
    def __init__(self, x, y, z, timestamp):
        self.history = []
        self.current_state = None

    def update(self, x, y, z, timestamp):
        if timestamp is None:
            timestamp = time.time()

        if self.current_state is not None:
            self.history = self.history + [self.current_state]
        
        self.current_state = {
            "timestamp": timestamp,
            "raw": {
                "x": x,
                "y": y,
                "z": z
            },
            "matched": None
        }
