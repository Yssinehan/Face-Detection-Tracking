"""Tests de validation de l'environnement de développement."""

import unittest
from pathlib import Path

import cv2
import mediapipe
import numpy

import config


class EnvironmentTestCase(unittest.TestCase):
    """Vérifie les prérequis locaux de la Phase 1."""

    def test_required_directories_exist(self):
        required_directories = (
            config.INPUT_VIDEO_DIRECTORY,
            config.OUTPUT_DIRECTORY,
            config.PROJECT_ROOT / "report",
            config.PROJECT_ROOT / "scripts",
            config.PROJECT_ROOT / "tests",
        )
        for directory in required_directories:
            with self.subTest(directory=directory):
                self.assertTrue(directory.is_dir())

    def test_configuration_values_are_available(self):
        self.assertTrue(config.PROJECT_NAME)
        self.assertIsInstance(config.PROJECT_ROOT, Path)
        self.assertIsInstance(config.INPUT_VIDEO_DIRECTORY, Path)
        self.assertIsInstance(config.OUTPUT_DIRECTORY, Path)

    def test_input_and_output_directories_are_different(self):
        self.assertNotEqual(
            config.INPUT_VIDEO_DIRECTORY.resolve(),
            config.OUTPUT_DIRECTORY.resolve(),
        )

    def test_supported_video_extensions_are_defined(self):
        self.assertEqual(
            config.SUPPORTED_VIDEO_EXTENSIONS,
            (".mp4", ".avi", ".mov"),
        )
        self.assertTrue(
            all(
                extension.startswith(".")
                for extension in config.SUPPORTED_VIDEO_EXTENSIONS
            )
        )

    def test_opencv_can_be_imported(self):
        self.assertTrue(cv2.__version__)

    def test_mediapipe_can_be_imported(self):
        self.assertTrue(mediapipe.__version__)

    def test_numpy_can_be_imported(self):
        self.assertTrue(numpy.__version__)


if __name__ == "__main__":
    unittest.main()
