"""Protect input videos and enforce inference tensor contracts."""

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from demo.gpu.client import decode_pose
from demo.gpu.run import replay


class PoseContractTests(unittest.TestCase):
    def test_pose_tensor_dtypes_are_not_interchangeable(self):
        for tensor in ("DETECTIONS", "KEYPOINTS"):
            with self.subTest(tensor=tensor):
                outputs = {
                    "DETECTIONS": np.zeros((1, 100, 6), np.float32),
                    "KEYPOINTS": np.zeros((1, 100, 17, 3), np.float32),
                    "NUM_DETECTIONS": np.array([[1]], np.int32),
                }
                outputs[tensor] = outputs[tensor].astype(np.int32)
                with self.assertRaises(ValueError):
                    decode_pose(outputs)


class InputPreservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_replaying_output_video_cannot_overwrite_the_input(self):
        class ControlledClient:
            def __init__(self, *args):
                self.count = 0

            def validate(self):
                return {"backend": "tensorrt", "instance_group": [{"kind": "KIND_GPU"}]}

            def stats(self):
                return {"model_stats": [{"inference_count": self.count}]}

            def infer(self, frame):
                self.count += 1
                return {"detections": []}

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "annotated.mp4"
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            self.assertTrue(writer.isOpened(), "Test requires the repository's mp4v video encoder")
            try:
                for value in (10, 80, 150, 220):
                    writer.write(np.full((48, 64, 3), value, np.uint8))
            finally:
                writer.release()
            original = target.read_bytes()
            args = argparse.Namespace(input=target, output=Path(directory), server_url="http://localhost",
                                      seconds=1, stride=1, max_width=128, concurrency=2, lookahead=2)
            with patch("demo.gpu.run.GpuPoseClient", ControlledClient):
                try:
                    await replay(args)
                except Exception:
                    # Every failure must preserve the caller's source, including validation failure.
                    pass
            self.assertEqual(target.read_bytes(), original, "An output path alias destroyed the source video")


if __name__ == "__main__":
    unittest.main()
