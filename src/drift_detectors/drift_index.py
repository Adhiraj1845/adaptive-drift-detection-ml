class DriftIndex:

    def __init__(self, detectors):
        self.detectors = detectors or []

    def compute_index(self, reference_data, new_data) -> float:
        if not self.detectors:
            return 0.0
        fired = sum(1 for d in self.detectors if d.detect(reference_data, new_data))
        return fired / len(self.detectors)
