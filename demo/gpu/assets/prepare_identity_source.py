"""Create an explicitly edited A/B/A video from two real NASA interviews.

Labels identify source segments only. No recognition identity, detection result,
or expected model score is painted into this input video.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
B_ID = 'jsc2021m000272_NASA_Astronaut_Candidate_Christina_Birch'
B_URL = f'https://images-assets.nasa.gov/video/{B_ID}/{B_ID}~medium.mp4'
GUIDELINES = 'https://www.nasa.gov/nasa-brand-center/images-and-media/'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(path):
    video = cv2.VideoCapture(str(path))
    fps = video.get(cv2.CAP_PROP_FPS)
    width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    while True:
        ok, _ = video.read()
        if not ok:
            break
        count += 1
    video.release()
    return {'sha256': sha(path), 'bytes': path.stat().st_size, 'width': width, 'height': height,
            'fps': fps, 'decoded_frames': count, 'duration_seconds': count / fps, 'audio': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/NotoSansSC-VF.ttf'))
    args = parser.parse_args()
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    prefix = cache / 'identity-b-medium-first-25MB.mp4'
    if not prefix.exists() or prefix.stat().st_size != 25_000_000:
        curl = shutil.which('curl.exe') or shutil.which('curl')
        if not curl:
            raise RuntimeError('curl is required to download the official video.')
        subprocess.run([curl, '-4', '--local-port', '19500-19600', '-f', '-sS', '-L',
                        '--range', '0-24999999', '--max-time', '120', B_URL, '-o', str(prefix)], check=True)
        if prefix.stat().st_size != 25_000_000:
            raise RuntimeError('Unexpected byte range size')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    a_dir = cache / 'source-a'
    a_path = a_dir / 'source-clip.mp4'
    a_manifest = a_dir / 'source-manifest.json'
    if not a_path.is_file() or not a_manifest.is_file():
        subprocess.run([sys.executable, '-B', str(ROOT / 'prepare_source.py'),
                        '--output-dir', str(a_dir)], check=True)
    a_source = json.loads(a_manifest.read_text(encoding='utf-8'))
    if sha(a_path) != a_source['output']['sha256']:
        raise RuntimeError('Source A does not match its provenance manifest')
    b_path = cache / 'source-b-clip.mp4'
    extracted = subprocess.run([ffmpeg, '-y', '-ss', '30', '-i', str(prefix), '-t', '15',
                                '-vf', 'scale=-2:480', '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
                                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(b_path)],
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
    (cache / 'identity-b-extract.log').write_text(extracted.stderr, encoding='utf-8')
    extracted.check_returncode()
    b_info = inspect(b_path)
    if (b_info['decoded_frames'], b_info['fps'], b_info['width'], b_info['height']) != (450, 30, 854, 480):
        raise RuntimeError(f'Unexpected B video: {b_info}')
    width, height, fps = 854, 480, 30
    font = ImageFont.truetype(str(args.font), 20)
    font2 = ImageFont.truetype(str(args.font), 17)
    for selected in (font, font2):
        try: selected.set_variation_by_axes([500])
        except OSError: pass
    bars = []
    for index, letter in [(1, 'A'), (2, 'B'), (3, 'A')]:
        bar = Image.new('RGB', (width, 58), '#0b141b')
        draw = ImageDraw.Draw(bar)
        draw.text((16, 2), f'EDITED SOURCE  {index}/3  |  SOURCE {letter}  |  NO IDENTITY RESULT IN INPUT',
                  font=font, fill='#55dfbb' if letter == 'A' else '#ffd177')
        draw.text((16, 31), '人工剪辑分段 · 非自然跨镜头跟踪 · 输入素材无预置身份确认结果', font=font2, fill='#e9f5f2')
        bars.append(np.asarray(bar)[:, :, ::-1].copy())
    output = ROOT / 'identity-source-clip.mp4'
    command = [ffmpeg, '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', 'pipe:0', '-c:v', 'libx264', '-crf', '20', '-preset', 'fast', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-an', str(output)]
    with (cache / 'identity-compose.log').open('wb') as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for segment, (path, first) in enumerate([(a_path, 0), (b_path, 0), (a_path, 150)]):
                reader = cv2.VideoCapture(str(path))
                reader.set(cv2.CAP_PROP_POS_FRAMES, first)
                try:
                    for _ in range(150):
                        ok, frame = reader.read()
                        if not ok:
                            raise RuntimeError('Source ended before segment completed')
                        frame[-58:] = bars[segment]
                        encoder.stdin.write(frame.tobytes())
                finally:
                    reader.release()
        finally:
            encoder.stdin.close()
            code = encoder.wait(timeout=90)
        if code:
            raise RuntimeError(f'Identity clip encoder failed: {code}')
    result = inspect(output)
    if result['decoded_frames'] != 450:
        raise RuntimeError('Montage must contain exactly 450 decoded real frames')
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(output), '-f', 'null', '-'],
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
    decoded.check_returncode()
    manifest = {
        'schema_version': 2, 'prepared_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        'purpose': 'Input for a face-enrollment / same-person confirmation / different-person rejection demonstration; not recognition output.',
        'source_a': {'label': 'Source A',
                     **{key: value for key, value in a_source.items() if key not in ('schema_version', 'prepared_at', 'output', 'reproduce')},
                     'original_excerpt_seconds': [20, 35],
                     'intermediate_clip': {'cache_only': True, **{key: value for key, value in a_source['output'].items() if key != 'path'}}},
        'source_b': {'label': 'Source B', 'publisher': 'NASA / Johnson Space Center (JSC)',
                     'title': 'NASA Astronaut Candidate Christina Birch interview (2021)', 'nasa_id': B_ID,
                     'individual_creator': 'Not named in NASA API metadata; credit NASA/JSC.', 'source_created': '2021-12-06',
                     'record_url': f'https://images.nasa.gov/details/{B_ID}', 'source_video_url': B_URL,
                     'collection_url': f'https://images-assets.nasa.gov/video/{B_ID}/collection.json',
                     'metadata_url': f'https://images-assets.nasa.gov/video/{B_ID}/metadata.json',
                     'original_excerpt_seconds': [30, 45], 'intermediate_clip': {'cache_only': True, **b_info},
                     'cache': {'byte_range_inclusive': [0, 24999999], 'complete_original': False, 'sha256_of_prefix_only': sha(prefix)}},
        'license': {'basis': 'NASA-produced informational media: generally not subject to US copyright per NASA guidelines; not a CC license.',
                    'official_guidelines': GUIDELINES, 'credit': 'NASA / Johnson Space Center',
                    'conditions': ['NASA and featured people do not endorse this project.',
                                   'NASA insignia, trademarks and publicity/privacy rights remain separate; not for product endorsement or commercial advertisements.',
                                   'No third-party copyright or music credit found in reviewed NASA metadata; audio removed from both excerpts.',
                                   'Video keeps NASA media terms; repository Apache-2.0 does not relicense it.']},
        'edit': {'format': 'A(5s) / B(5s) / A(5s)', 'explicitly_edited': True,
                 'not_natural_camera_transition': True, 'static_slides': False, 'synthetic_motion': False,
                 'source_labels_burned_in': True, 'identity_predictions_burned_in': False,
                 'segments': [
                     {'frames': [0, 149], 'timeline_seconds': [0, 5], 'source': 'A', 'source_original_seconds': [20, 25]},
                     {'frames': [150, 299], 'timeline_seconds': [5, 10], 'source': 'B', 'source_original_seconds': [30, 35]},
                     {'frames': [300, 449], 'timeline_seconds': [10, 15], 'source': 'A', 'source_original_seconds': [25, 30]}]},
        'evaluation_policy': {'identities': 'Use anonymous labels such as Demo-P001; real names are attribution metadata only.',
                              'reference_frames': 'Use frame 0 of this final video for Source A and frame 150 for Source B; no separate hand-picked gallery is required.',
                              'expectations': ['Enroll Source A from its first real frame.',
                                               'Confirm Source A only when actual face similarity and association checks pass.',
                                               'Source B must not be positively matched to the enrolled A identity; leave unknown unless separately enrolled.',
                                               'Reappearance of A is a face-confirmation test, not natural cross-camera tracking.',
                                               'No face means only a body track / pending confirmation; a pose box is not identity evidence.']},
        'output': {'path': 'identity-source-clip.mp4', **result, 'full_decode_exit_code': decoded.returncode},
        'reproduce': 'python -B demo/gpu/assets/prepare_identity_source.py --font /path/to/NotoSansSC.ttf'
    }
    (ROOT / 'identity-source-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'source_b': b_info, 'identity_montage': result}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
