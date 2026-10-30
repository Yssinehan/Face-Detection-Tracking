"""Suivi géométrique multi-visages sans identification biométrique."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import hypot

from detector import FaceDetection

BoundingBox = tuple[int, int, int, int]
LOGGER = logging.getLogger(__name__)


def _valid_box(box: BoundingBox) -> bool:
    return len(box) == 4 and box[2] > 0 and box[3] > 0


def calculate_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calcule l'intersection sur union de deux boîtes."""
    if not _valid_box(box_a) or not _valid_box(box_b):
        return 0.0
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def calculate_centroid_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Calcule la distance euclidienne entre les centroïdes."""
    if not _valid_box(box_a) or not _valid_box(box_b):
        return float("inf")
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    return hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))


@dataclass
class Track:
    """État mutable d'une piste pendant une session."""

    track_id: int
    bbox: BoundingBox
    centroid: tuple[float, float]
    confidence: float
    age: int = 1
    hits: int = 1
    missing_frames: int = 0
    missing_seconds: float = 0.0
    last_seen_frame: int = 1
    last_seen_time: float = 0.0
    previous_centroid: tuple[float, float] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    state: str = "tentative"
    detected: bool = True


def _centroid(box: BoundingBox) -> tuple[float, float]:
    return (box[0] + box[2] / 2, box[1] + box[3] / 2)


class FaceTracker:
    """Association gloutonne IoU/centroïde avec cycle de vie temporel."""

    def __init__(
        self,
        max_missing_frames: int = 10,
        min_track_hits: int = 2,
        max_centroid_distance: float = 180.0,
        min_iou: float = 0.0,
        association_threshold: float = 1.25,
        alpha: float = 0.6,
        beta: float = 0.4,
        max_missing_seconds: float | None = None,
        debug: bool = False,
    ) -> None:
        if max_missing_frames < 0 or min_track_hits < 1:
            raise ValueError("Les paramètres de cycle de vie doivent être positifs.")
        if max_missing_seconds is not None and max_missing_seconds <= 0:
            raise ValueError("max_missing_seconds doit être positif.")
        if not 0 <= min_iou <= 1 or association_threshold <= 0:
            raise ValueError("Les seuils du tracker sont invalides.")
        if alpha < 0 or beta < 0 or alpha + beta == 0:
            raise ValueError("Les poids d'association sont invalides.")
        self.max_missing_frames = max_missing_frames
        self.max_missing_seconds = max_missing_seconds
        self.min_track_hits = min_track_hits
        self.max_centroid_distance = max_centroid_distance
        self.min_iou = min_iou
        self.association_threshold = association_threshold
        self.alpha = alpha / (alpha + beta)
        self.beta = beta / (alpha + beta)
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        """Réinitialise une session et son compteur d'identifiants."""
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._frame_number = 0
        self._clock = 0.0

    def update(
        self,
        detections: list[FaceDetection],
        frame_shape: tuple[int, ...] | None = None,
        timestamp: float | None = None,
    ) -> list[Track]:
        """Met à jour les pistes et renvoie les pistes conservées."""
        del frame_shape
        self._frame_number += 1
        self._clock = (
            float(timestamp)
            if timestamp is not None and timestamp >= self._clock
            else self._clock + 1.0
        )
        valid_detections = self._deduplicate_detections(
            [
                detection for detection in detections
                if _valid_box(self._detection_box(detection))
            ]
        )
        self._log(
            "raw frame=%d count=%d detections=%s",
            self._frame_number,
            len(detections),
            [
                (d.x, d.y, d.width, d.height, round(d.confidence, 3))
                for d in detections
            ],
        )
        self._log(
            "filtered frame=%d count=%d detections=%s",
            self._frame_number,
            len(valid_detections),
            [
                (d.x, d.y, d.width, d.height, round(d.confidence, 3))
                for d in valid_detections
            ],
        )
        predicted_boxes = {
            track_id: self._predicted_box(track)
            for track_id, track in self._tracks.items()
        }
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(valid_detections):
                detection_box = self._detection_box(detection)
                distance = calculate_centroid_distance(
                    predicted_boxes[track_id], detection_box
                )
                iou = calculate_iou(predicted_boxes[track_id], detection_box)
                size_ratio = self._size_ratio(track.bbox, detection_box)
                distance_gate = self.max_centroid_distance * max(
                    1, min(track.missing_frames + 1, 5)
                )
                normalized_distance = distance / max(distance_gate, 1.0)
                size_penalty = min(abs(1.0 - size_ratio), 1.0)
                cost = (
                    self.alpha * normalized_distance
                    + self.beta * (1 - iou)
                    + 0.15 * size_penalty
                )
                if (
                    distance <= distance_gate
                    and size_ratio <= 2.5
                    and cost <= self.association_threshold
                ):
                    candidates.append((cost, track_id, detection_index))
                    self._log(
                        "candidate frame=%d track=%d detection=%d cost=%.3f "
                        "distance=%.1f iou=%.3f",
                        self._frame_number,
                        track_id,
                        detection_index,
                        cost,
                        distance,
                        iou,
                    )

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for cost, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            self._update_track(self._tracks[track_id], valid_detections[detection_index])
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            self._log(
                "association frame=%d track=%d detection=%d cost=%.3f",
                self._frame_number,
                track_id,
                detection_index,
                cost,
            )

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.age += 1
            track.missing_frames += 1
            track.missing_seconds = max(0.0, self._clock - track.last_seen_time)
            track.detected = False
            track.state = "missing"
            expired_by_frames = track.missing_frames > self.max_missing_frames
            expired_by_time = (
                self.max_missing_seconds is not None
                and track.missing_seconds > self.max_missing_seconds
            )
            if expired_by_frames or expired_by_time:
                self._log(
                    "delete frame=%d track=%d missing_frames=%d missing_seconds=%.2f",
                    self._frame_number,
                    track_id,
                    track.missing_frames,
                    track.missing_seconds,
                )
                del self._tracks[track_id]

        for detection_index, detection in enumerate(valid_detections):
            if detection_index not in matched_detections:
                self._register_track(detection)

        self._remove_redundant_tracks(matched_tracks)
        self._log_state()
        return self.get_active_tracks()

    def get_active_tracks(self) -> list[Track]:
        """Renvoie toutes les pistes retenues, triées par identifiant."""
        return [self._tracks[key] for key in sorted(self._tracks)]

    def _remove_redundant_tracks(self, matched_tracks: set[int]) -> None:
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks or track.detected:
                continue
            if any(
                calculate_iou(track.bbox, self._tracks[matched_id].bbox) >= 0.5
                for matched_id in matched_tracks
                if matched_id in self._tracks
            ):
                self._log("delete redundant track=%d", track_id)
                del self._tracks[track_id]

    @staticmethod
    def _detection_box(detection: FaceDetection) -> BoundingBox:
        return (detection.x, detection.y, detection.width, detection.height)

    @staticmethod
    def _size_ratio(box_a: BoundingBox, box_b: BoundingBox) -> float:
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        if area_a <= 0 or area_b <= 0:
            return float("inf")
        return max(area_a / area_b, area_b / area_a)

    @staticmethod
    def _predicted_box(track: Track) -> BoundingBox:
        x, y, width, height = track.bbox
        steps = min(max(1, track.missing_frames + 1), 5)
        return (
            round(x + track.velocity[0] * steps),
            round(y + track.velocity[1] * steps),
            width,
            height,
        )

    def _register_track(self, detection: FaceDetection) -> None:
        box = self._detection_box(detection)
        state = "confirmed" if self.min_track_hits == 1 else "tentative"
        self._tracks[self._next_id] = Track(
            track_id=self._next_id,
            bbox=box,
            centroid=_centroid(box),
            confidence=detection.confidence,
            state=state,
            last_seen_frame=self._frame_number,
            last_seen_time=self._clock,
        )
        self._log("create frame=%d track=%d", self._frame_number, self._next_id)
        self._next_id += 1

    def _update_track(self, track: Track, detection: FaceDetection) -> None:
        box = self._detection_box(detection)
        new_centroid = _centroid(box)
        if track.previous_centroid is not None:
            measured_velocity = (
                new_centroid[0] - track.centroid[0],
                new_centroid[1] - track.centroid[1],
            )
            track.velocity = (
                0.7 * track.velocity[0] + 0.3 * measured_velocity[0],
                0.7 * track.velocity[1] + 0.3 * measured_velocity[1],
            )
        track.previous_centroid = track.centroid
        track.bbox = box
        track.centroid = new_centroid
        track.confidence = detection.confidence
        track.age += 1
        track.hits += 1
        track.missing_frames = 0
        track.missing_seconds = 0.0
        track.last_seen_frame = self._frame_number
        track.last_seen_time = self._clock
        track.detected = True
        if track.hits >= self.min_track_hits:
            track.state = "confirmed"

    @staticmethod
    def _deduplicate_detections(
        detections: list[FaceDetection],
    ) -> list[FaceDetection]:
        """Applique une NMS légère et déterministe avec IoU 0.5."""
        selected: list[FaceDetection] = []
        for detection in sorted(
            detections, key=lambda item: item.confidence, reverse=True
        ):
            box = (
                detection.x,
                detection.y,
                detection.width,
                detection.height,
            )
            if all(
                calculate_iou(
                    box,
                    (other.x, other.y, other.width, other.height),
                )
                < 0.5
                for other in selected
            ):
                selected.append(detection)
        return selected

    def _log(self, message: str, *args: object) -> None:
        if self.debug:
            LOGGER.info(message, *args)

    def _log_state(self) -> None:
        self._log(
            "state frame=%d tracks=%s confirmed=%s missing=%s",
            self._frame_number,
            sorted(self._tracks),
            sorted(
                track_id
                for track_id, track in self._tracks.items()
                if track.state == "confirmed" and track.detected
            ),
            sorted(
                track_id
                for track_id, track in self._tracks.items()
                if track.state == "missing"
            ),
        )
