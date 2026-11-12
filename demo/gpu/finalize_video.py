"""Make the actual GPU output browser-playable and attach verifiable evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matching_source_manifest(path, input_sha256):
    if not path.is_file():
        return None
    source = json.loads(path.read_text(encoding='utf-8'))
    return source if source.get('output', {}).get('sha256') == input_sha256 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'identity-run-output')
    parser.add_argument('--container', default='vf-open-gpu-identity-triton-gpu-identity-1')
    parser.add_argument('--source-manifest',type=Path,default=HERE/'assets/identity-source-manifest.json')
    parser.add_argument('--model-manifest',type=Path,default=ROOT.parent/'visionflow-inference/gpu_demo/identity-model-manifest.json')
    args = parser.parse_args()
    directory = args.output.resolve()
    receipt = directory/'manifest.json'
    meta = json.loads(receipt.read_text(encoding='utf-8'))
    # Resolve all server evidence before changing any existing output. A stopped
    # or mistyped container must leave the previous video and receipt intact.
    gpu = subprocess.run(['docker','exec',args.container,'nvidia-smi',
        '--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True)
    image = subprocess.run(['docker','inspect',args.container,'--format','{{.Image}}'],
                           capture_output=True,text=True,check=True)
    expected = meta['output']['frames']
    video = directory/'annotated.mp4'
    candidate = directory/'annotated.browser.mp4'
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    conversion = subprocess.run([ffmpeg,'-y','-v','error','-i',str(video),'-an','-c:v','libx264',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(candidate)],capture_output=True,text=True)
    conversion.check_returncode()
    decode = subprocess.run([ffmpeg,'-v','error','-i',str(candidate),'-f','null','-'],capture_output=True,text=True)
    decode.check_returncode()
    capture = cv2.VideoCapture(str(candidate))
    count = 0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        if count in (0, expected//2, expected-1):
            name=f'preview-{count:03d}.png'
            if not cv2.imwrite(str(directory/name),frame):
                raise RuntimeError('Preview image write failed')
            frames.append(name)
        count += 1
    capture.release()
    if count != expected:
        raise RuntimeError('Browser video does not match the number of GPU-inferred frames')
    # From this point a failed publication cannot leave a stale success receipt.
    receipt.unlink()
    os.replace(candidate,video)
    meta['output']['codec']='H.264 / yuv420p / MP4, faststart'
    meta['verification']={'full_decode_exit_code':decode.returncode,'decoded_frames':count}
    # This identity is queried from the actual server container, separately from
    # any client-host GPU. No container env or credentials are collected.
    meta['server_device']={'gpu':gpu.stdout.strip(),'image_id':image.stdout.strip(),
        'evidence':'Read-only nvidia-smi executed inside the same isolated Triton container'}
    model_manifest=args.model_manifest
    if model_manifest.is_file():
        shutil.copyfile(model_manifest,directory/'model-manifest.json')
    source_manifest=args.source_manifest
    meta['source'].pop('media_manifest_sha256', None)
    if matching_source_manifest(source_manifest, meta['source']['sha256']) is not None:
        meta['source']['media_manifest_sha256']=digest(source_manifest)
    meta['code_sha256']={str(path.relative_to(ROOT)).replace('\\','/'):digest(path)
        for path in [HERE/'client.py',HERE/'run.py',HERE/'finalize_video.py',
                     ROOT/'core/gpu_video_pipeline.py',ROOT/'core/ordered_pipeline.py']}
    if meta.get('identity',{}).get('enabled'):
        for path in [HERE/'identity_client.py',ROOT/'core/identity_gallery.py']:
            meta['code_sha256'][str(path.relative_to(ROOT)).replace('\\','/')]=digest(path)
    for name in ['annotated.mp4','model-manifest.json',*frames]:
        path=directory/name
        if path.is_file(): meta['assets'][name]={'sha256':digest(path),'bytes':path.stat().st_size}
    pending_receipt=receipt.with_suffix('.json.tmp')
    pending_receipt.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(pending_receipt,receipt)
    print(json.dumps({'run_id':meta['run_id'],'video_frames':count,'server_gpu':gpu.stdout.strip(),
                      'format':meta['output']['codec']},indent=2))


if __name__=='__main__': main()
