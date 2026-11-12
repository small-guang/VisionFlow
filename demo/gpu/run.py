"""Replay a real video through the existing Triton GPU pose ensemble.

Every output frame is inferred, recorded and committed in input order. The
side-by-side video is an actual inference artifact, not a slideshow or a screen
recording. Kafka/Milvus are not used by this bounded replay entry point.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.gpu_video_pipeline import process_video_frames
from demo.gpu.client import GpuPoseClient, NATIVE_MODEL, ENSEMBLE

EDGES = ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
         (11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def draw_result(image, result):
    painted = image.copy()
    for detection in result['detections']:
        x1,y1,x2,y2 = [round(x) for x in detection['box']]
        cv2.rectangle(painted,(x1,y1),(x2,y2),(55,225,170),2,cv2.LINE_AA)
        label = f"person {detection['score']:.2f}"
        cv2.putText(painted,label,(max(0,x1),max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,(55,225,170),1,cv2.LINE_AA)
        identity=detection.get('identity')
        if identity:
            identifier=identity['identity_id'] or identity['candidate_id'] or 'unknown'
            state=identity['status'].replace('_',' ')
            cv2.putText(painted,f'{identifier} / {state}',(max(0,x1),min(image.shape[0]-8,y1+20)),
                        cv2.FONT_HERSHEY_SIMPLEX,.46,(255,235,90),1,cv2.LINE_AA)
            similarity=identity.get('face_similarity')
            if similarity is not None:
                cv2.putText(painted,f'face cos {similarity:.3f} / gate 0.500',
                            (max(0,x1),min(image.shape[0]-8,y1+39)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,235,90),1,cv2.LINE_AA)
            body_similarity=identity.get('body_similarity')
            if body_similarity is not None:
                cv2.putText(painted,f'body cos {body_similarity:.3f} / candidate only',
                            (max(0,x1),min(image.shape[0]-8,y1+57)),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,235,90),1,cv2.LINE_AA)
            if detection.get('face'):
                fx1,fy1,fx2,fy2=map(round,detection['face']['box'])
                cv2.rectangle(painted,(fx1,fy1),(fx2,fy2),(255,170,55),1,cv2.LINE_AA)
        points = np.asarray(detection['keypoints'])
        for a,b in EDGES:
            if min(points[a,2],points[b,2]) >= .4:
                cv2.line(painted,tuple(np.rint(points[a,:2]).astype(int)),
                         tuple(np.rint(points[b,:2]).astype(int)),(40,210,255),2,cv2.LINE_AA)
        for x,y,score in points:
            if score >= .4:
                cv2.circle(painted,(round(x),round(y)),3,(255,170,55),-1,cv2.LINE_AA)
    return painted


def compose_frame(frame, result, index, source_fps, gallery_images=None):
    height,width = frame.shape[:2]
    extra=240 if result.get('identity_enabled') else 0
    canvas = np.full((height+100,width*2+extra,3),(26,21,13),np.uint8)
    canvas[55:55+height,:width] = frame
    canvas[55:55+height,width:width*2] = draw_result(frame,result)
    labels = [('SOURCE / real continuous footage',(15,32)),
              ('GPU / face + person identity' if result.get('identity_enabled') else 'GPU / YOLO26m pose + Triton',(width+15,32)),
              (f'frame {index:04d} | t={index/source_fps:.2f}s | {len(result["detections"])} person(s)',(15,height+84))]
    for text,xy in labels:
        cv2.putText(canvas,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.58,(230,241,235),1,cv2.LINE_AA)
    if extra:
        left=width*2+12
        cv2.putText(canvas,'ENROLLED THIS RUN',(left,32),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,241,235),1,cv2.LINE_AA)
        for number,(identity,images) in enumerate((gallery_images or {}).items()):
            if number>=2: break
            y=58+number*175
            cv2.putText(canvas,identity,(left,y+17),cv2.FONT_HERSHEY_SIMPLEX,.5,(55,225,170),1,cv2.LINE_AA)
            for offset,key in ((0,'face'),(108,'body')):
                crop=images.get(key)
                if crop is not None and crop.size:
                    canvas[y+28:y+124,left+offset:left+offset+96]=cv2.resize(crop,(96,96))
                cv2.putText(canvas,key,(left+offset,y+142),cv2.FONT_HERSHEY_SIMPLEX,.4,(230,241,235),1,cv2.LINE_AA)
    return canvas


def count_inferences(stats):
    return sum(int(item.get('inference_count',0)) for item in stats.get('model_stats',[]))


async def replay(args):
    output = args.output.resolve()
    source = args.input.resolve()
    # Resolve symlinks before any output mutation. Never truncate an input video
    # that happens to have the generated output filename.
    if source == (output/'annotated.mp4').resolve():
        raise ValueError('Input video and generated video must be different files')
    if not source.is_file():
        raise ValueError('Input video does not exist')
    if not 1 <= args.concurrency <= 32 or not 0 <= args.lookahead <= 32:
        raise ValueError('concurrency must be 1..32; lookahead must be 0..32')
    output.mkdir(parents=True,exist_ok=True)
    # Remove the success receipt first so a failed rerun cannot masquerade as an old success.
    (output/'manifest.json').unlink(missing_ok=True)
    identity_enabled=getattr(args,'identity',False)
    gallery=None
    if identity_enabled:
        from core.identity_gallery import IdentityGallery
        from demo.gpu.identity_client import GpuIdentityClient
        client=GpuIdentityClient(args.server_url,body_model=getattr(args,'body_model',None))
        gallery=IdentityGallery()
    else:
        client = GpuPoseClient(args.server_url)
    config = client.validate()
    before = client.stats()
    identity_before={name:client.json_get(f'/v2/models/{name}/stats')
                     for name in getattr(client,'identity_configs',{})}
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError('Cannot decode input video')
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or not 0 < source_fps <= 240:
        capture.release()
        raise ValueError('Source video FPS must be finite and within 0..240')
    fps = source_fps/args.stride
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(width,height) < 1:
        capture.release()
        raise ValueError('Invalid video dimensions')
    scale = min(1.,args.max_width/width)
    width,height = max(2,int(width*scale)//2*2),max(2,int(height*scale)//2*2)
    destination = output/'annotated.mp4'
    output_width=width*2+(240 if identity_enabled else 0)
    writer = cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(output_width,height+100))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError('Cannot initialize video encoder')
    run_id = 'gpu-'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
    frame_hashes, gallery_images = set(), {}
    max_frames = round(args.seconds*source_fps)
    def frames():
        index = 0
        while index < max_frames:
            ok,frame = capture.read()
            if not ok: break
            current = index
            index += 1
            if current % args.stride: continue
            frame = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
            frame_hashes.add(hashlib.sha256(frame.tobytes()).hexdigest())
            yield current, frame

    # Offload synchronous HTTP; the shared ordered pipeline bounds real requests
    # and retains frames until ordered output, rather than accumulating futures.
    executor = ThreadPoolExecutor(max_workers=args.concurrency,thread_name_prefix='gpu-rpc')
    loop = asyncio.get_running_loop()
    events_path = output/'events.jsonl'
    started = datetime.now(timezone(timedelta(hours=8))).isoformat()
    try:
        with events_path.open('w',encoding='utf-8') as log:
            async def infer(frame):
                return await loop.run_in_executor(executor,client.infer,frame)
            async def commit(index,frame,result):
                if gallery is not None:
                    result['identity_enabled']=True
                    for person in result['detections']:
                        person['identity']=gallery.observe(face_embedding=person.get('face_embedding'),
                                                          body_embedding=person.get('body_embedding'))
                        decision=person['identity']
                        if decision['status']=='enrolled_face' and person.get('face'):
                            def crop(box):
                                x1,y1,x2,y2=map(round,box)
                                return frame[max(0,y1):min(height,y2),max(0,x1):min(width,x2)].copy()
                            gallery_images[decision['identity_id']]={'face':crop(person['face']['box']),
                                                                    'body':crop(person['box'])}
                event = {'run_id':run_id,'frame_index':index,'timestamp_seconds':index/source_fps,
                         'model':ENSEMBLE,**result}
                log.write(json.dumps(event,separators=(',',':'))+'\n')
                painted = compose_frame(frame,result,index,source_fps,gallery_images)
                writer.write(painted)
            summary = await process_video_frames(frames(),infer,commit,
                concurrency=args.concurrency,lookahead=args.lookahead)
    finally:
        capture.release()
        writer.release()
        # Let in-flight HTTP calls end before cleanup can claim success.
        executor.shutdown(wait=True,cancel_futures=True)
    after = client.stats()
    identity_after={name:client.json_get(f'/v2/models/{name}/stats') for name in identity_before}
    delta = count_inferences(after)-count_inferences(before)
    if not summary['frames'] or delta < summary['frames']:
        raise RuntimeError('No complete GPU inference evidence for output frames')
    if len(frame_hashes)<2:
        raise RuntimeError('Input lacks distinct video frames; refusing a slideshow artifact')
    check = cv2.VideoCapture(str(destination))
    decoded = 0
    while True:
        ok,_ = check.read()
        if not ok: break
        decoded += 1
    check.release()
    if decoded != summary['frames']:
        raise RuntimeError('Encoded video frame count does not match committed GPU results')
    try:
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
                            capture_output=True,text=True,check=True).stdout.strip()
    except (FileNotFoundError,subprocess.CalledProcessError):
        gpu = None
    manifest = {'schema_version':1,'run_id':run_id,'started_at':started,'client_host_gpu':gpu,
        'device_evidence_scope':'nvidia-smi is from the client host; server GPU identity must be recorded separately for remote servers.',
        'scope':'Continuous video replay through existing Triton GPU pose ensemble; no Kafka/Milvus run is claimed.',
        'inference_backend':config,'models':{'ensemble':ENSEMBLE,'native':NATIVE_MODEL},
        'source':{'name':args.input.name,'sha256':sha256(args.input),'fps':source_fps,
                  'stride':args.stride,'distinct_decoded_frames':len(frame_hashes)},
        'summary':summary,'gpu_inference_count_delta':delta,'stats_before':before,'stats_after':after,
        'output':{'fps':fps,'frames':decoded,'duration_seconds':decoded/fps,
                  'width':output_width,'height':height+100,'codec':'MPEG-4 Part 2 (transcode to H.264 for browser)'},
        'timing_scope':'Client await infer wall time includes RPC, CPU pre/postprocess and GPU execution; not GPU kernel time.',
        'assets':{name:{'sha256':sha256(output/name),'bytes':(output/name).stat().st_size}
                  for name in ['annotated.mp4','events.jsonl']}}
    if gallery is not None:
        manifest['identity']={'enabled':True,'gallery_size':gallery.size,'face_threshold':gallery.face_threshold,
            'body_threshold':gallery.body_threshold,'confirmation_policy':'face-confirmed; body-only is a candidate, never an independent identity confirmation',
            'configs':client.identity_configs,
            'gpu_inference_deltas':{name:count_inferences(identity_after[name])-count_inferences(identity_before[name])
                                    for name in identity_before},
            'stats_before':identity_before,'stats_after':identity_after,
            'embedding_spaces':{
                'face':'SFace 128D; ArcFace 5-point 112x112 similarity alignment; RGB FP32 0..255; gallery L2',
                'body':'OSNet-AIN 512D; OpenCV INTER_LINEAR 128x256 RGB; /255; ImageNet mean/std; gallery L2'
                        if getattr(args,'body_model',None) else None},
            'scope':'Anonymous in-memory demonstration gallery, not real-name recognition or a calibrated production identity system.'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run_id':run_id,'frames':decoded,'gpu_inferences':delta,'output':str(destination)},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'identity-run-output')
    parser.add_argument('--server-url',default='http://127.0.0.1:18010')
    parser.add_argument('--seconds',type=float,default=15)
    parser.add_argument('--stride',type=int,default=2)
    parser.add_argument('--max-width',type=int,default=640)
    parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--lookahead',type=int,default=2)
    parser.add_argument('--identity',action='store_true',default=True,help='Use the face/person identity pipeline (enabled by default)')
    parser.add_argument('--body-model',default='osnet_gpu',help='Licensed GPU body encoder model name')
    args=parser.parse_args()
    if not 0<args.seconds<=120 or not 1<=args.stride<=30 or not 128<=args.max_width<=1280 or not 1<=args.concurrency<=32 or not 0<=args.lookahead<=32:
        parser.error('seconds 0..120, stride 1..30, max-width 128..1280, concurrency 1..32, lookahead 0..32 required')
    asyncio.run(replay(args))


if __name__=='__main__': main()
