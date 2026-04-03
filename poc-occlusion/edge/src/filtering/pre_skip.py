class PreFrameSkipper:
    def __init__(self, skip_rate: int = 2):
        self.skip_rate = max(1, skip_rate)

    def should_process(self, frame_idx: int) -> bool:
        return frame_idx % self.skip_rate == 0
