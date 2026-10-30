"""Tests sans matériel pour la gestion de capture webcam."""

import unittest
from unittest.mock import patch

from camera_processor import process_camera


class FakeCamera:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


class CameraTestCase(unittest.TestCase):
    def test_unavailable_camera_is_rejected(self):
        camera = FakeCamera([])
        camera.isOpened = lambda: False
        with self.assertRaises(OSError):
            process_camera(None, camera=camera)
        self.assertTrue(camera.released)

    def test_capture_failure_releases_camera(self):
        camera = FakeCamera([])
        with patch("camera_processor.cv2.destroyAllWindows"):
            with self.assertRaises(OSError):
                process_camera(None, camera=camera)
        self.assertTrue(camera.released)

    def test_camera_index_validation_is_in_cli(self):
        from app import main

        with self.assertRaises(SystemExit):
            main(["--camera", "--camera-index", "-1"])


if __name__ == "__main__":
    unittest.main()
