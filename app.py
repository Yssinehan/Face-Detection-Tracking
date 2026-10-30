"""Interface en ligne de commande de la Phase 2."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import config
from camera_processor import process_camera
from detector import FaceDetector, load_image
from video_processor import process_video
from visualization import draw_detections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Détection de visages MediaPipe sur image ou vidéo."
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--image", type=Path, help="Chemin de l'image à traiter.")
    inputs.add_argument("--video", type=Path, help="Chemin de la vidéo à traiter.")
    inputs.add_argument("--camera", action="store_true", help="Utiliser la webcam.")
    parser.add_argument("--output", type=Path, help="Chemin de sortie personnalisé.")
    parser.add_argument("--display", action="store_true", help="Afficher les frames.")
    parser.add_argument("--overwrite", action="store_true", help="Remplacer la sortie.")
    parser.add_argument("--track", action="store_true", help="Activer le suivi vidéo.")
    parser.add_argument("--camera-index", type=int, default=0, help="Index webcam.")
    parser.add_argument("--record", action="store_true", help="Enregistrer la webcam.")
    parser.add_argument("--confidence", type=float, help="Seuil de confiance.")
    return parser


def process_image(
    input_path: Path,
    output_path: Path,
    detector: FaceDetector,
    *,
    overwrite: bool = False,
) -> int:
    if output_path.suffix.lower() not in config.SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Extension de sortie image non supportée : {output_path.suffix}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"La sortie existe déjà : {output_path}")
    image = load_image(input_path)
    detections = detector.detect(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), draw_detections(image, detections)):
        raise OSError(f"Impossible d'écrire l'image : {output_path}")
    print(f"Visages détectés : {len(detections)}")
    print(f"Image annotée : {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.image is None and args.video is None and not args.camera:
        parser.print_help()
        return 0
    if args.camera and args.output is not None and not args.record:
        parser.error("--output avec --camera nécessite --record.")
    if args.camera and args.track:
        parser.error("--track est implicite avec --camera.")
    if args.record and not args.camera:
        parser.error("--record nécessite --camera.")
    if args.camera_index < 0:
        parser.error("--camera-index doit être positif ou nul.")
    try:
        output_path = args.output or (
            config.DEFAULT_IMAGE_OUTPUT_PATH
            if args.image is not None
            else config.DEFAULT_VIDEO_OUTPUT_PATH
        )
        threshold = (
            config.CONFIDENCE_THRESHOLD
            if args.confidence is None
            else args.confidence
        )
        if not 0.0 <= threshold <= 1.0:
            parser.error("--confidence doit être compris entre 0 et 1.")
        with FaceDetector(confidence_threshold=threshold) as detector:
            if args.camera:
                stats = process_camera(
                    detector,
                    camera_index=args.camera_index,
                    record=args.record,
                    output_path=args.output,
                )
                print(
                    f"Frames webcam : {stats.frames_processed}; "
                    f"FPS de traitement : {stats.processing_fps:.2f}"
                )
                if stats.output_path is not None:
                    print(f"Webcam enregistrée : {stats.output_path}")
                return 0
            if args.image is not None:
                if args.display:
                    parser.error("--display n'est disponible que pour une vidéo.")
                return process_image(
                    args.image, output_path, detector, overwrite=args.overwrite
                )
            stats = process_video(
                args.video,
                output_path,
                detector,
                display=args.display,
                overwrite=args.overwrite,
                track=args.track,
            )
        print(
            f"Frames traitées : {stats.frames_processed}; "
            f"FPS de traitement : {stats.processing_fps:.2f}"
        )
        print(f"Vidéo annotée : {stats.output_path}")
        return 0
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Erreur : {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
