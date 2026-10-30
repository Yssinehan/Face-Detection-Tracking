"""Valide l'environnement et la structure du projet de Phase 1."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_DIRECTORIES = (
    "videos",
    "images",
    "output",
    "models",
    "tests",
    "scripts",
    "report",
)
REQUIRED_FILES = (
    "app.py",
    "detector.py",
    "tracker.py",
    "video_processor.py",
    "config.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    ".env.example",
    "visualization.py",
    "tests/__init__.py",
    "tests/test_environment.py",
)


def check_python() -> tuple[bool, str]:
    version = sys.version_info
    valid = version >= (3, 10)
    return valid, f"Python: {sys.version.split()[0]} ({sys.executable})"


def check_virtual_environment() -> tuple[bool, str]:
    virtual_env = sys.prefix != sys.base_prefix
    return virtual_env, "Virtual Environment: " + ("OK" if virtual_env else "NON")


def check_import(module_name: str, display_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        return False, f"{display_name}: NON ({error})"
    version = getattr(module, "__version__", "version non exposée")
    return True, f"{display_name}: OK ({version})"


def check_project_structure() -> tuple[bool, str]:
    missing_directories = [
        path for path in REQUIRED_DIRECTORIES
        if not (PROJECT_ROOT / path).is_dir()
    ]
    missing_files = [
        path for path in REQUIRED_FILES
        if not (PROJECT_ROOT / path).is_file()
    ]
    missing = missing_directories + missing_files
    if missing:
        return False, f"Project Structure: NON (éléments absents: {', '.join(missing)})"
    return True, "Project Structure: OK"


def check_configuration() -> tuple[bool, str]:
    try:
        import config

        paths_valid = (
            config.PROJECT_ROOT == PROJECT_ROOT
            and config.INPUT_VIDEO_DIRECTORY == PROJECT_ROOT / "videos"
            and config.INPUT_IMAGE_DIRECTORY == PROJECT_ROOT / "images"
            and config.OUTPUT_DIRECTORY == PROJECT_ROOT / "output"
            and config.MODEL_PATH.is_file()
            and config.INPUT_VIDEO_DIRECTORY != config.OUTPUT_DIRECTORY
        )
        extensions_valid = all(
            isinstance(extension, str) and extension.startswith(".")
            for extension in config.SUPPORTED_VIDEO_EXTENSIONS
        )
    except (ImportError, AttributeError, TypeError):
        return False, "Configuration: NON (importation ou valeurs invalides)"
    if not paths_valid or not extensions_valid:
        return False, "Configuration: NON (chemins, modèle ou extensions invalides)"
    return True, "Configuration: OK"


def main() -> int:
    """Exécute toutes les validations et retourne un code shell."""
    print("=" * 40)
    print("AI FACE DETECTION & TRACKING")
    print("ENVIRONMENT VALIDATION")
    print("=" * 40)

    checks = [
        check_python(),
        check_virtual_environment(),
        check_import("cv2", "OpenCV"),
        check_import("mediapipe", "MediaPipe"),
        check_import("numpy", "NumPy"),
        check_project_structure(),
        check_configuration(),
    ]
    for _, message in checks:
        print(message)

    print("\n" + "=" * 40)
    if all(passed for passed, _ in checks):
        print("ENVIRONMENT STATUS: READY")
        print("=" * 40)
        return 0
    print("ENVIRONMENT STATUS: NOT READY")
    print("=" * 40)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
