"""Run the separate VisionFlow CPU face detection demonstration."""
import argparse
import base64
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np


def detect(url, image_bytes):
    body = json.dumps({"image_base64": base64.b64encode(image_bytes).decode("ascii")}).encode("utf-8")
    request = Request(url.rstrip("/") + "/detect", data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description="CPU demo: send an image to VisionFlow Inference and save annotated output")
    parser.add_argument("image", type=Path, help="input image; use a file you have permission to share")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="CPU demo service URL")
    parser.add_argument("--output", type=Path, default=Path("demo-output"), help="output directory")
    args = parser.parse_args()
    try:
        image_bytes = args.image.read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("input is not a supported image")
        result = detect(args.url, image_bytes)
        for detection in result["detections"]:
            x, y, width, height = detection["box"]
            cv2.rectangle(image, (x, y), (x + width, y + height), (54, 218, 110), 2)
            cv2.putText(image, "face", (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (54, 218, 110), 2)
        args.output.mkdir(parents=True, exist_ok=True)
        output_image = args.output / "detected.jpg"
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("could not encode output image")
        output_image.write_bytes(encoded.tobytes())
        (args.output / "detections.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Found {len(result['detections'])} candidate(s). Saved {output_image} and detections.json")
    except (OSError, ValueError, KeyError, HTTPError, URLError) as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
