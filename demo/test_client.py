import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from client import detect


class ClientTests(unittest.TestCase):
    def test_client_sends_image_and_receives_detection(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert payload["image_base64"]
                data = json.dumps({"image_size": [20, 10], "detections": [{"box": [1, 2, 3, 4], "score": 0.5}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            ok, image = cv2.imencode(".png", np.zeros((10, 20, 3), dtype=np.uint8))
            self.assertTrue(ok)
            result = detect(f"http://127.0.0.1:{server.server_port}", image.tobytes())
            self.assertEqual(result["detections"][0]["box"], [1, 2, 3, 4])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
