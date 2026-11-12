"""Fetch an official NASA video prefix and extract one real 15-second interview shot.

The source shot and its provenance stay in an ignored cache. The identity
builder embeds this provenance in the final A/B/A media manifest.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import argparse
import json
import shutil
import subprocess
import urllib.parse
import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
NASA_ID = 'jsc2021m000284_NASA_Astronaut_Candidate_Marcos_Berrios - Spanish Language'
ENCODED_ID = urllib.parse.quote(NASA_ID, safe='')
BASE = f'https://images-assets.nasa.gov/video/{ENCODED_ID}/{ENCODED_ID}'
SOURCE_URL = BASE + '~medium.mp4'
LICENSE_URL = 'https://www.nasa.gov/nasa-brand-center/images-and-media/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.cache' / 'source-a',
                        help='Intermediate clip and manifest directory (default: ignored .cache/source-a)')
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    prefix = cache / 'nasa-medium-first-15MB.mp4'
    if not prefix.exists() or prefix.stat().st_size != 15_000_000:
        curl = shutil.which('curl.exe') or shutil.which('curl')
        if not curl:
            raise RuntimeError('curl is required to retrieve the official public video.')
        subprocess.run([curl, '-4', '--local-port', '18500-18600', '-f', '-sS', '-L',
                        '--range', '0-14999999', '--max-time', '120', SOURCE_URL, '-o', str(prefix)], check=True)
        if prefix.stat().st_size != 15_000_000:
            raise RuntimeError('NASA source did not return the expected 15MB byte range.')
    output = output_dir / 'source-clip.mp4'
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, '-y', '-ss', '20', '-i', str(prefix), '-t', '15', '-vf', 'scale=-2:480',
               '-c:v', 'libx264', '-crf', '20', '-preset', 'fast', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-an', str(output)]
    run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (cache / 'source-extract.log').write_text(run.stderr, encoding='utf-8')
    run.check_returncode()
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(output), '-f', 'null', '-'],
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
    decoded.check_returncode()
    reader = cv2.VideoCapture(str(output))
    fps = reader.get(cv2.CAP_PROP_FPS)
    width, height = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = 0
    previous = None
    motion = []
    import numpy as np
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        if previous is not None:
            motion.append(float(np.abs(frame.astype(np.float32) - previous.astype(np.float32)).mean()))
        previous = frame
        frames += 1
    reader.release()
    if (frames, fps, width, height) != (450, 30, 854, 480):
        raise RuntimeError(f'Unexpected clip metadata: {(frames, fps, width, height)}')
    manifest = {
        'schema_version': 1,
        'prepared_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'title': 'NASA Astronaut Candidate Marcos Berrios — Spanish Language interview (2021)',
        'nasa_id': NASA_ID,
        'publisher': 'NASA / Johnson Space Center (JSC)',
        'individual_creator': 'Not named in NASA API metadata; credit NASA/JSC.',
        'source_created': '2021-12-06',
        'record_url': f'https://images.nasa.gov/details/{ENCODED_ID}',
        'collection_url': f'https://images-assets.nasa.gov/video/{ENCODED_ID}/collection.json',
        'metadata_url': f'https://images-assets.nasa.gov/video/{ENCODED_ID}/metadata.json',
        'source_video_url': SOURCE_URL,
        'license': {
            'basis': 'NASA-produced informational media. NASA guidelines state its media generally are not subject to copyright in the United States. This is not a CC license.',
            'official_guidelines': LICENSE_URL,
            'third_party_notice_review': 'NASA/JSC API metadata identifies this as an astronaut-candidate interview; no separate third-party copyright or music credit was found in that metadata. Audio is removed.',
            'conditions': ['Credit NASA/JSC as source.', 'Do not imply NASA or the featured person endorses this project.',
                           'NASA insignia, trademarks, privacy and publicity rights are separate from copyright; do not repurpose this as a product endorsement or commercial advertisement.',
                           'Third-party material, if separately identified, retains its own rights.'],
            'project_relicensing': 'The video is not relicensed under the repository Apache-2.0 license.'
        },
        'source_download': {
            'cached_byte_range_inclusive': [0, 14999999],
            'cached_bytes': prefix.stat().st_size,
            'cached_prefix_sha256': digest(prefix),
            'complete_download': False,
            'full_remote_object_bytes_from_head': 111991280,
            'source_encoding': '1280x720, H.264 Main, yuv420p, 30 fps (NASA medium derivative).',
            'note': 'Only the cached prefix is hashed; no full-original checksum is claimed. Cache files are excluded from Git.'
        },
        'edit': {
            'source_start_seconds': 20,
            'source_end_seconds_exclusive': 35,
            'duration_seconds': 15,
            'operations': ['Select one continuous 15-second interview segment.', 'Downscale 1280x720 to 854x480.',
                           'Encode H.264 yuv420p MP4 with faststart.', 'Remove audio.'],
            'no_synthetic_motion': True,
            'no_generated_boxes_in_source': True,
            'selection_note': 'Selected after inspecting source frames; excludes the source film question/title cards near 12-16s and 40s.'
        },
        'output': {'path': 'source-clip.mp4', 'sha256': digest(output), 'bytes': output.stat().st_size,
                   'width': width, 'height': height, 'fps': fps, 'decoded_frames': frames,
                   'duration_seconds': frames / fps, 'audio': False,
                   'full_decode_exit_code': decoded.returncode,
                   'mean_adjacent_frame_pixel_difference': round(float(np.mean(motion)), 6),
                   'max_adjacent_frame_pixel_difference': round(float(np.max(motion)), 6)},
        'reproduce': 'python -B demo/gpu/assets/prepare_source.py',
    }
    (output_dir / 'source-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest['output'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
