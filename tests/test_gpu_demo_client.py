import unittest
import json
import numpy as np

from demo.gpu.client import build_request, parse_response, decode_pose, require_gpu_config


class ProtocolTests(unittest.TestCase):
    def test_image_bytes_remain_bgr_with_explicit_batch_and_binary_size(self):
        frame = np.array([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8)
        body, length = build_request(frame)
        meta = json.loads(body[:length])
        self.assertEqual(meta['inputs'][0]['shape'], [1, 1, 2, 3])
        self.assertEqual(meta['inputs'][0]['parameters']['binary_data_size'], 6)
        self.assertEqual(body[length:], bytes([11, 22, 33, 44, 55, 66]))

    def test_binary_response_rejects_truncated_tensor(self):
        meta = {'outputs': [{'name': 'NUM_DETECTIONS', 'datatype': 'INT32', 'shape': [1, 1],
                             'parameters': {'binary_data_size': 4}}]}
        header = json.dumps(meta).encode()
        with self.assertRaises(ValueError):
            parse_response(header + b'\x01\x00', len(header))

    def test_binary_response_decodes_little_endian_int(self):
        header = json.dumps({'outputs': [{'name': 'NUM_DETECTIONS', 'datatype': 'INT32',
            'shape': [1, 1], 'parameters': {'binary_data_size': 4}}]}).encode()
        result = parse_response(header + b'\x02\x00\x00\x00', len(header))
        self.assertEqual(result['NUM_DETECTIONS'].tolist(), [[2]])

    def test_pose_count_bounds_and_finite_values_are_enforced(self):
        outputs = {'DETECTIONS': np.zeros((1, 100, 6), np.float32),
                   'KEYPOINTS': np.zeros((1, 100, 17, 3), np.float32),
                   'NUM_DETECTIONS': np.array([[101]], np.int32)}
        with self.assertRaises(ValueError): decode_pose(outputs)
        outputs['NUM_DETECTIONS'][0, 0] = 1
        outputs['DETECTIONS'][0, 0] = [1, 2, 10, 20, .75, 0]
        result = decode_pose(outputs)
        self.assertEqual(result['detections'][0]['box'], [1., 2., 10., 20.])
        self.assertEqual(result['detections'][0]['score'], .75)
        outputs['KEYPOINTS'][0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError): decode_pose(outputs)

    def test_cpu_or_missing_instance_is_rejected(self):
        for config in [{}, {'backend':'onnxruntime', 'instance_group':[{'kind':'KIND_CPU'}]}]:
            with self.assertRaises(ValueError): require_gpu_config(config)
        require_gpu_config({'backend':'tensorrt','instance_group':[{'kind':'KIND_GPU'}]})

if __name__ == '__main__': unittest.main()
