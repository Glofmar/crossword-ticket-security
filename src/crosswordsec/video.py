from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    index: int
    time_s: float
    image: np.ndarray


class VideoSampler:
    """Iterate over a video, one frame every `step_s` seconds."""

    def __init__(self, path, step_s):
        self.path = Path(path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"cannot open {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise IOError(f"no frame rate in {self.path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.step = max(1, round(step_s * self.fps))

    @property
    def duration_s(self):
        return self.n_frames / self.fps

    def __iter__(self):
        # grab() through the skipped frames instead of seeking: seeking is not
        # frame-accurate on every codec and the steps are small anyway
        index = 0
        while True:
            ok, image = self.cap.read()
            if not ok:
                return
            yield Frame(index, index / self.fps, image)
            index += 1
            for _ in range(self.step - 1):
                if not self.cap.grab():
                    return
                index += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
