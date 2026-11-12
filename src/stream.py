"""Frame source. Works the same for a video file or an rtsp:// URL."""
import time
import cv2


class FrameSource:
    def __init__(self, uri, realtime=False):
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {uri}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.realtime = realtime  # pace a file like a live camera
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def __iter__(self):
        i, t0 = 0, time.time()
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.realtime:
                wait = i / self.fps - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
            yield i, i / self.fps, frame
            i += 1
        self.cap.release()
