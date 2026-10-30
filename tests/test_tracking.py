"""Tests déterministes du tracker géométrique."""

import unittest

from detector import FaceDetection
from tracker import FaceTracker, calculate_centroid_distance, calculate_iou


def face(x, y, width=40, height=40, confidence=0.9):
    return FaceDetection(x, y, width, height, confidence)


class TrackingTestCase(unittest.TestCase):
    def test_iou_cases(self):
        self.assertEqual(calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(calculate_iou((0, 0, 10, 10), (20, 20, 10, 10)), 0.0)
        self.assertAlmostEqual(calculate_iou((0, 0, 10, 10), (5, 0, 10, 10)), 1 / 3)
        self.assertEqual(calculate_iou((0, 0, 0, 10), (0, 0, 10, 10)), 0.0)

    def test_centroid_distance(self):
        self.assertEqual(calculate_centroid_distance((0, 0, 10, 10), (0, 0, 10, 10)), 0)
        self.assertAlmostEqual(
            calculate_centroid_distance((0, 0, 10, 10), (10, 0, 10, 10)), 10
        )

    def test_empty_and_single_face(self):
        tracker = FaceTracker()
        self.assertEqual(tracker.update([]), [])
        tracks = tracker.update([face(10, 10)])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)

    def test_persistent_id_and_motion(self):
        tracker = FaceTracker()
        first = tracker.update([face(10, 10)])
        second = tracker.update([face(25, 10)])
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(second[0].bbox[:2], (25, 10))
        self.assertEqual(second[0].state, "confirmed")

    def test_two_faces_are_one_to_one(self):
        tracker = FaceTracker()
        tracks = tracker.update([face(10, 10), face(200, 10)])
        self.assertEqual({track.track_id for track in tracks}, {1, 2})
        tracks = tracker.update([face(20, 10), face(190, 10)])
        self.assertEqual({track.track_id for track in tracks}, {1, 2})

    def test_missing_and_reappearance_preserve_id(self):
        tracker = FaceTracker(max_missing_frames=2)
        first = tracker.update([face(10, 10)])[0]
        tracker.update([])
        missing = tracker.update([])[0]
        self.assertEqual(missing.track_id, first.track_id)
        self.assertFalse(missing.detected)
        reappeared = tracker.update([face(20, 10)])[0]
        self.assertEqual(reappeared.track_id, first.track_id)
        self.assertTrue(reappeared.detected)

    def test_longer_gap_uses_motion_prediction(self):
        tracker = FaceTracker(max_missing_frames=5, max_centroid_distance=50)
        first = tracker.update([face(10, 10)])[0]
        tracker.update([face(30, 10)])
        tracker.update([])
        tracker.update([])
        reappeared = tracker.update([face(90, 10)])[0]
        self.assertEqual(reappeared.track_id, first.track_id)

    def test_webcam_length_gap_preserves_id(self):
        tracker = FaceTracker(max_missing_frames=180, max_centroid_distance=250)
        first = tracker.update([face(100, 100)])[0]
        for _ in range(120):
            tracker.update([])
        reappeared = tracker.update([face(120, 110)])[0]
        self.assertEqual(reappeared.track_id, first.track_id)

    def test_time_expiration_overrides_large_frame_budget(self):
        tracker = FaceTracker(
            max_missing_frames=180,
            max_centroid_distance=100,
            max_missing_seconds=3.0,
        )
        first = tracker.update([face(100, 100)], timestamp=0.0)[0]
        tracker.update([], timestamp=2.9)
        tracker.update([], timestamp=3.1)
        self.assertEqual(tracker.get_active_tracks(), [])
        replacement = tracker.update([face(100, 100)], timestamp=3.2)[0]
        self.assertNotEqual(replacement.track_id, first.track_id)

    def test_far_reappearance_gets_new_id(self):
        tracker = FaceTracker(max_missing_frames=5, max_centroid_distance=50)
        first = tracker.update([face(10, 10)])[0]
        tracker.update([])
        replacement = next(
            track for track in tracker.update([face(300, 10)]) if track.detected
        )
        self.assertNotEqual(replacement.track_id, first.track_id)

    def test_nearby_faces_remain_distinct(self):
        tracker = FaceTracker(max_centroid_distance=100)
        tracks = tracker.update([face(10, 10), face(65, 10)])
        self.assertEqual(len(tracks), 2)
        updated = tracker.update([face(20, 10), face(75, 10)])
        self.assertEqual({track.track_id for track in updated}, {1, 2})

    def test_expiry_assigns_new_id(self):
        tracker = FaceTracker(max_missing_frames=1)
        first = tracker.update([face(10, 10)])[0]
        tracker.update([])
        tracker.update([])
        self.assertEqual(len(tracker.get_active_tracks()), 0)
        new = tracker.update([face(10, 10)])[0]
        self.assertNotEqual(new.track_id, first.track_id)

    def test_duplicate_detections_do_not_share_a_track(self):
        tracker = FaceTracker()
        tracks = tracker.update([face(10, 10), face(11, 11)])
        self.assertEqual(len({track.track_id for track in tracks}), 1)
        tracks = tracker.update([face(12, 12)])
        self.assertEqual(len({track.track_id for track in tracks}), 1)

    def test_reset_clears_state(self):
        tracker = FaceTracker()
        tracker.update([face(10, 10)])
        tracker.reset()
        self.assertEqual(tracker.get_active_tracks(), [])
        self.assertEqual(tracker.update([face(10, 10)])[0].track_id, 1)


if __name__ == "__main__":
    unittest.main()
