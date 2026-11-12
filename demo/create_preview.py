"""Create the README preview from an actual CPU demo run."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("assets/astronaut.png"))
    parser.add_argument("--result", type=Path, default=Path("run-output/detected.jpg"))
    parser.add_argument("--json", type=Path, default=Path("run-output/detections.json"))
    parser.add_argument("--output", type=Path, default=Path("assets/cpu-face-result.png"))
    args = parser.parse_args()
    source, result = read_image(args.input), read_image(args.result)
    count = len(json.loads(args.json.read_text(encoding="utf-8"))["detections"])
    canvas = np.full((820, 1440, 3), (25, 20, 17), dtype=np.uint8)
    cv2.putText(canvas, "VISIONFLOW / CPU DEMO", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (237, 234, 227), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Original image", (60, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (185, 175, 165), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Face detection result", (750, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (185, 175, 165), 1, cv2.LINE_AA)
    canvas[155:731, 60:636] = cv2.resize(source, (576, 576))
    canvas[155:731, 750:1326] = cv2.resize(result, (576, 576))
    cv2.putText(canvas, f"{count} candidate(s)  |  OpenCV Haar cascade  |  CPU", (60, 780),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 225, 175), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Image: NASA / scikit-image", (995, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (175, 170, 165), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Failed to encode preview")
    args.output.write_bytes(data.tobytes())
    print(args.output)


if __name__ == "__main__":
    main()
