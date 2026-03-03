class CUSUMDetector:
    """
    Simple CUSUM for concept drift.
    """

    def __init__(self, threshold: float = 5.0, drift: float = 0.01):
        self.threshold = threshold
        self.drift = drift
        self.reset()

    def reset(self):
        self.pos_sum = 0.0
        self.neg_sum = 0.0

    def update(self, x: float) -> bool:
        self.pos_sum = max(0, self.pos_sum + x - self.drift)
        self.neg_sum = min(0, self.neg_sum + x + self.drift)

        if self.pos_sum > self.threshold or abs(self.neg_sum) > self.threshold:
            self.reset()
            return True
        return False
