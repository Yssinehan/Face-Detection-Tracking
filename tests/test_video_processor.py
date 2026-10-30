"""Tests du pipeline vidéo avec un détecteur simulé."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_processor import process_video


class StubDetector:
    """Détecteur minimal permettant de tester le pipeline indépendamment."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        del frame
        self.calls += 1
        return []


def create_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("Codec mp4v indisponible dans cette installation OpenCV.")
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class VideoProcessorTestCase(unittest.TestCase):
    def test_missing_and_unsupported_videos_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                process_video(root / "missing.mp4", root / "out.mp4", StubDetector())
            unsupported = root / "input.txt"
            unsupported.write_text("not a video", encoding="utf-8")
            with self.assertRaises(ValueError):
                process_video(unsupported, root / "out.mp4", StubDetector())

    def test_video_is_processed_and_output_dimensions_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            create_video(input_path)
            detector = StubDetector()
            stats = process_video(input_path, output_path, detector)
            self.assertEqual(stats.frames_processed, 3)
            self.assertEqual(detector.calls, 3)
            self.assertTrue(output_path.is_file())
            capture = cv2.VideoCapture(str(output_path))
            try:
                self.assertTrue(capture.isOpened())
                success, frame = capture.read()
                self.assertTrue(success)
                self.assertEqual(frame.shape[:2], (48, 64))
            finally:
                capture.release()

    def test_existing_output_requires_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.mp4"
            output_path = root / "output.mp4"
            create_video(input_path)
            output_path.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                process_video(input_path, output_path, StubDetector())

    def test_empty_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "empty.mp4"
            create_video(input_path, frame_count=0)
            with self.assertRaises(ValueError):
                process_video(input_path, root / "out.mp4", StubDetector())


if __name__ == "__main__":
    unittest.main()
