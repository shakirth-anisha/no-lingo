import cv2


class CameraCapture:
    def __init__(self, use_raspi=False, cam_index=0, frame_size=(1640, 1232)):
        self.use_raspi = use_raspi
        self.cap = None
        self.picam2 = None
        self._opened = False

        if self.use_raspi:
            try:
                from picamera2 import Picamera2
            except ImportError as exc:
                raise ImportError(
                    "Picamera2 is required when using --raspi. Install it with: sudo apt install python3-picamera2"
                ) from exc

            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": "BGR888", "size": frame_size}
            )
            self.picam2.configure(config)
            self.picam2.start()
            self._opened = True
        else:
            self.cap = cv2.VideoCapture(cam_index)
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
                self._opened = self.cap.isOpened()

    def isOpened(self):
        if self.use_raspi:
            return self._opened
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if self.use_raspi:
            if not self._opened:
                return False, None
            frame = self.picam2.capture_array()
            return True, frame
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.use_raspi:
            if self.picam2 is not None:
                self.picam2.stop()
            self._opened = False
            return
        if self.cap is not None:
            self.cap.release()
