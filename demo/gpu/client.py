"""Triton V2 binary HTTP client for the real GPU pose ensemble.

No CPU inference implementation or implicit backend fallback is provided.
Pre/postprocessing belongs to the existing Triton ensemble.
"""
from __future__ import annotations

import http.client
import json
from urllib.parse import urlsplit

import numpy as np

ENSEMBLE = 'yolo26m_pose_ensemble'
NATIVE_MODEL = 'yolo26m_pose_onnx'
OUTPUTS = ('DETECTIONS', 'KEYPOINTS', 'NUM_DETECTIONS')
DTYPES = {'FP32': np.dtype('<f4'), 'INT32': np.dtype('<i4')}


def build_request(bgr):
    image = np.asarray(bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a BGR uint8 [H,W,3] image')
    h, w = image.shape[:2]
    if min(h, w) < 1 or max(h, w) > 4096 or h * w > 4_194_304:
        raise ValueError('Image dimensions exceed demo limits')
    raw = np.ascontiguousarray(image).tobytes()
    meta = {'inputs': [{'name': 'RAW_IMAGE', 'datatype': 'UINT8', 'shape': [1, h, w, 3],
                       'parameters': {'binary_data_size': len(raw)}}],
            'outputs': [{'name': name, 'parameters': {'binary_data': True}} for name in OUTPUTS]}
    header = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    return header + raw, len(header)


def parse_response(body, header_length):
    if not isinstance(header_length, int) or not 0 < header_length <= len(body):
        raise ValueError('Invalid Triton binary header length')
    meta = json.loads(body[:header_length])
    if 'error' in meta:
        raise RuntimeError(meta['error'])
    offset, result = header_length, {}
    for output in meta.get('outputs', []):
        name, shape = output['name'], output['shape']
        dtype = DTYPES.get(output['datatype'])
        if dtype is None or not shape or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('Invalid output datatype or shape')
        count = 1
        for dimension in shape:
            count *= dimension
        size = output.get('parameters', {}).get('binary_data_size')
        if size != count * dtype.itemsize or size > 16_777_216 or offset + size > len(body):
            raise ValueError('Truncated or inconsistent Triton output')
        if name in result:
            raise ValueError('Duplicate Triton output')
        result[name] = np.frombuffer(body[offset:offset + size], dtype=dtype).reshape(shape).copy()
        offset += size
    if offset != len(body):
        raise ValueError('Unexpected trailing response bytes')
    return result


def decode_pose(outputs):
    boxes, points, counts = (outputs[name] for name in OUTPUTS)
    if boxes.shape != (1, 100, 6) or points.shape != (1, 100, 17, 3) or counts.shape != (1, 1):
        raise ValueError('Pose ensemble output contract mismatch')
    if counts.dtype != np.int32:
        raise ValueError('Invalid detection count dtype')
    if boxes.dtype != np.float32 or points.dtype != np.float32:
        raise ValueError('Detection coordinates and keypoints must be FP32')
    n = int(counts[0, 0])
    if not 0 <= n <= 100 or not np.isfinite(boxes[:,:n]).all() or not np.isfinite(points[:,:n]).all():
        raise ValueError('Invalid detection count or nonfinite output')
    return {'detections': [{'box': row[:4].tolist(), 'score': float(row[4]),
                            'class_id': int(row[5]), 'keypoints': kp.tolist()}
                           for row, kp in zip(boxes[0, :n], points[0, :n])]}


def require_gpu_config(config):
    groups = config.get('instance_group', [])
    if config.get('backend') not in ('tensorrt', 'onnxruntime') or not groups:
        raise ValueError('Native GPU model configuration is required')
    if any(group.get('kind') != 'KIND_GPU' for group in groups):
        raise ValueError('Refusing a CPU/automatic inference instance')


class GpuPoseClient:
    def __init__(self, server_url='http://127.0.0.1:18000', timeout=60):
        self.url = urlsplit(server_url)
        if self.url.scheme not in ('http', 'https') or not self.url.hostname or self.url.path not in ('', '/'):
            raise ValueError('Server URL must contain scheme and host, without a path')
        self.timeout = timeout

    def request(self, path, body=None, headers=None):
        cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        connection = cls(self.url.hostname, self.url.port, timeout=self.timeout)
        try:
            connection.request('POST' if body is not None else 'GET', path, body=body, headers=headers or {})
            response = connection.getresponse()
            # Demo output is capped; input and response never share a global unbounded queue.
            data = response.read(16_777_217)
            if len(data) > 16_777_216:
                raise ValueError('Triton response exceeds limit')
            if response.status != 200:
                raise RuntimeError(f'Triton HTTP {response.status}: {data[:500]!r}')
            return data, response.getheader('Inference-Header-Content-Length')
        finally:
            connection.close()

    def json_get(self, path):
        return json.loads(self.request(path)[0])

    def validate(self):
        config = self.json_get(f'/v2/models/{NATIVE_MODEL}/config')
        require_gpu_config(config)
        self.request(f'/v2/models/{ENSEMBLE}/ready')
        return config

    def stats(self):
        return self.json_get(f'/v2/models/{NATIVE_MODEL}/stats')

    def infer(self, bgr):
        body, length = build_request(bgr)
        response, header = self.request(f'/v2/models/{ENSEMBLE}/infer', body,
            {'Content-Type': 'application/octet-stream', 'Inference-Header-Content-Length': str(length)})
        if header is None:
            raise ValueError('Server did not return binary output metadata')
        return decode_pose(parse_response(response, int(header)))

    def infer_arrays(self, model, inputs, output_names):
        chunks, descriptors = [], []
        for name, array in inputs.items():
            array=np.ascontiguousarray(array)
            datatype={np.dtype('float32'):'FP32',np.dtype('uint8'):'UINT8'}.get(array.dtype)
            if datatype is None or array.size==0 or array.nbytes>16_777_216:
                raise ValueError('Unsupported input tensor')
            raw=array.tobytes()
            descriptors.append({'name':name,'shape':list(array.shape),'datatype':datatype,
                                'parameters':{'binary_data_size':len(raw)}})
            chunks.append(raw)
        meta={'inputs':descriptors,'outputs':[{'name':name,'parameters':{'binary_data':True}}
                                             for name in output_names]}
        header=json.dumps(meta,separators=(',',':')).encode()
        response,length=self.request(f'/v2/models/{model}/infer',header+b''.join(chunks),
            {'Content-Type':'application/octet-stream','Inference-Header-Content-Length':str(len(header))})
        if length is None: raise ValueError('Missing binary metadata')
        return parse_response(response,int(length))
