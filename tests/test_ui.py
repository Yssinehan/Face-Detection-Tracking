"""Tests unitaires des helpers UI, sans navigateur ni webcam physique."""

import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import av

from ui.image_page import _decode_upload
from ui.video_page import _save_upload, _video_info
from ui.webcam_page import WebcamStats, WebcamVideoProcessor


class Upload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class UiHelpersTestCase(unittest.TestCase):
    def test_webcam_stats_are_immutable(self):
        stats = WebcamStats(frames=1, detected_faces=1)
        with self.assertRaises((AttributeError, TypeError)):
            stats.frames = 2

    def test_webcam_processor_reuses_detector_and_tracker(self):
        class FakeDetector:
            def __init__(self):
                self.calls = 0
                self.closed = False

            def detect(self, image):
                self.calls += 1
                return []

            def close(self):
                self.closed = True

        class FakeTracker:
            def __init__(self):
                self.calls = 0

            def update(self, detections, frame_shape, timestamp):
                self.calls += 1
                return []

        detector = FakeDetector()
        tracker = FakeTracker()
        processor = WebcamVideoProcessor(
            detector_factory=lambda: detector,
            tracker_factory=lambda: tracker,
        )
        self.assertIsNotNone(processor)
        processor.on_ended()
        self.assertTrue(detector.closed)

    def test_webcam_processor_converts_frame_and_updates_stats(self):
        class FakeDetector:
            def detect(self, image):
                self.shape = image.shape
                return []

            def close(self):
                pass

        class FakeTracker:
            def update(self, detections, frame_shape, timestamp):
                self.shape = frame_shape
                return []

        processor = WebcamVideoProcessor(FakeDetector, FakeTracker)
        frame = av.VideoFrame.from_ndarray(
            np.zeros((16, 24, 3), dtype=np.uint8),
            format="bgr24",
        )
        output = processor.recv(frame)
        self.assertEqual(output.format.name, "bgr24")
        self.assertEqual(processor.get_stats().frames, 1)
        processor.on_ended()

    def test_webcam_sessions_do_not_share_tracker_state(self):
        class FakeDetector:
            def detect(self, image):
                return []

            def close(self):
                pass

        class FakeTracker:
            def __init__(self):
                self.calls = 0

            def update(self, detections, frame_shape, timestamp):
                self.calls += 1
                return []

        first = WebcamVideoProcessor(FakeDetector, FakeTracker)
        second = WebcamVideoProcessor(FakeDetector, FakeTracker)
        frame = av.VideoFrame.from_ndarray(
            np.zeros((8, 8, 3), dtype=np.uint8),
            format="bgr24",
        )
        first.recv(frame)
        self.assertEqual(first.get_stats().frames, 1)
        self.assertEqual(second.get_stats().frames, 0)
        first.on_ended()
        second.on_ended()

    def test_image_validation_rejects_empty_file(self):
        with self.assertRaises(ValueError):
            _decode_upload(Upload("empty.png", b""))

    def test_image_validation_rejects_invalid_bytes(self):
        with self.assertRaises(ValueError):
            _decode_upload(Upload("broken.png", b"not-an-image"))

    def test_image_validation_decodes_valid_png(self):
        ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(ok)
        image = _decode_upload(Upload("valid.png", encoded.tobytes()))
        self.assertEqual(image.shape, (8, 8, 3))

    def test_video_upload_rejects_unsupported_extension(self):
        with self.assertRaises(ValueError):
            _save_upload(Upload("video.txt", b"data"))

    def test_video_upload_rejects_empty_file(self):
        with self.assertRaises(ValueError):
            _save_upload(Upload("video.mp4", b""))

    def test_video_upload_uses_safe_temporary_path(self):
        source = _save_upload(Upload("../unsafe.mp4", b"video-data"))
        try:
            self.assertTrue(source.is_file())
            self.assertEqual(source.suffix, ".mp4")
            self.assertNotEqual(source.name, "../unsafe.mp4")
        finally:
            source.unlink(missing_ok=True)

    def test_video_info_rejects_invalid_video(self):
        with NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
            path = Path(temporary.name)
        path.write_bytes(b"not-a-video")
        try:
            with self.assertRaises(ValueError):
                capture, _ = _video_info(path)
                capture.release()
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
