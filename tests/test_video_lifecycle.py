"""Video lifecycle checks with model/video dependencies replaced by fakes."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

cv = types.SimpleNamespace(
    CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2,
    CAP_PROP_FPS=3, CAP_PROP_FRAME_COUNT=4,
    VideoCapture=Mock(), VideoWriter=Mock(), VideoWriter_fourcc=Mock(return_value=0),
)
with patch.dict(sys.modules, {
    'cv2': cv, 'mediapipe': types.ModuleType('mediapipe'),
    'ultralytics': types.SimpleNamespace(YOLO=Mock()),
    'yt_dlp': types.ModuleType('yt_dlp'),
}):
    from src.analyzer import TennisProAnalyzer


class VideoLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.source = Path(self.folder.name) / 'input.mp4'
        self.source.write_bytes(b'original-video-fixture')
        self.output = Path(self.folder.name) / 'output.mp4'
        self.analyzer = TennisProAnalyzer('fixture.pt')
        self.analyzer.pose_detector = Mock()
        self.analyzer.initialize_models = Mock()
        self.analyzer.download_video = Mock(side_effect=AssertionError('Local input was downloaded'))
        self.capture = Mock()
        self.capture.isOpened.return_value = True
        properties = {1: 640, 2: 480, 3: 29.97, 4: 1}
        self.capture.get.side_effect = properties.__getitem__
        self.writer = Mock()
        self.writer.isOpened.return_value = True
        cv.VideoCapture.reset_mock()
        cv.VideoWriter.reset_mock()
        cv.VideoCapture.return_value = self.capture
        cv.VideoWriter.return_value = self.writer

    def test_same_input_output_is_rejected_without_modifying_source(self):
        with self.assertRaises(ValueError):
            self.analyzer.process_video(str(self.source), str(self.source))
        self.assertEqual(self.source.read_bytes(), b'original-video-fixture')
        cv.VideoCapture.assert_not_called()

    def test_invalid_video_closes_capture_and_pose(self):
        self.capture.isOpened.return_value = False
        with self.assertRaises(ValueError):
            self.analyzer.process_video(str(self.source), str(self.output))
        self.capture.release.assert_called_once()
        self.analyzer.pose_detector.close.assert_called_once()
        cv.VideoWriter.assert_not_called()

    def test_failed_writer_closes_both_video_resources_and_pose(self):
        self.writer.isOpened.return_value = False
        with self.assertRaises(OSError):
            self.analyzer.process_video(str(self.source), str(self.output))
        self.capture.release.assert_called_once()
        self.writer.release.assert_called_once()
        self.analyzer.pose_detector.close.assert_called_once()

    def test_decode_failure_closes_resources_and_preserves_fractional_fps(self):
        self.capture.read.side_effect = RuntimeError('fixture decoder failure')
        with self.assertRaisesRegex(RuntimeError, 'fixture decoder failure'):
            self.analyzer.process_video(str(self.source), str(self.output))
        self.capture.release.assert_called_once()
        self.writer.release.assert_called_once()
        self.analyzer.pose_detector.close.assert_called_once()
        self.assertEqual(cv.VideoWriter.call_args.args[2], 29.97)
        self.analyzer.download_video.assert_not_called()


if __name__ == '__main__':
    unittest.main()
