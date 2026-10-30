"""Tests unitaires du détecteur et des validations d'images."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import config
from detector import FaceDetection, FaceDetector, _clip_box, load_image


class DetectorTestCase(unittest.TestCase):
    """Couvre les entrées invalides et le format des résultats."""

    @classmethod
    def setUpClass(cls):
        cls.detector = FaceDetector()

    @classmethod
    def tearDownClass(cls):
        cls.detector.close()

    def test_valid_blank_image_returns_empty_results(self):
        results = self.detector.detect(np.zeros((100, 120, 3), dtype=np.uint8))
        self.assertEqual(results, [])

    def test_invalid_and_empty_images_are_rejected(self):
        with self.assertRaises(TypeError):
            self.detector.detect("not-an-image")
        with self.assertRaises(ValueError):
            self.detector.detect(np.empty((0, 0, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((20, 20), dtype=np.uint8))

    def test_bounding_box_is_clipped_to_image(self):
        self.assertEqual(_clip_box(-10, -5, 40, 30, 100, 80), (0, 0, 30, 25))
        self.assertIsNone(_clip_box(50, 50, 0, 20, 100, 80))

    def test_detection_result_structure(self):
        detection = FaceDetection(1, 2, 30, 40, 0.95)
        self.assertEqual((detection.x, detection.y), (1, 2))
        self.assertGreaterEqual(detection.confidence, 0)
        self.assertLessEqual(detection.confidence, 1)

    def test_confidence_threshold_and_multiple_results(self):
        class Category:
            def __init__(self, score):
                self.score = score

        class Box:
            def __init__(self, x, y, width, height):
                self.origin_x = x
                self.origin_y = y
                self.width = width
                self.height = height

        class Detection:
            def __init__(self, score, box):
                self.categories = [Category(score)]
                self.bounding_box = box

        class Result:
            detections = [
                Detection(0.9, Box(1, 2, 20, 20)),
                Detection(0.2, Box(30, 2, 20, 20)),
            ]

        class FakeModel:
            def detect(self, image):
                del image
                return Result()

        original_model = self.detector._detector
        self.detector._detector = FakeModel()
        try:
            results = self.detector.detect(np.zeros((40, 60, 3), dtype=np.uint8))
        finally:
            self.detector._detector = original_model
        self.assertEqual(results, [FaceDetection(1, 2, 20, 20, 0.9)])

    def test_load_image_validates_path_and_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            cv2.imwrite(str(path), np.zeros((10, 10, 3), dtype=np.uint8))
            self.assertEqual(load_image(path).shape, (10, 10, 3))
            with self.assertRaises(FileNotFoundError):
                load_image(Path(directory) / "missing.png")
            unsupported = Path(directory) / "image.txt"
            unsupported.write_text("invalid", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_image(unsupported)

    def test_model_asset_exists(self):
        self.assertTrue(config.MODEL_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
