"""GPU face detection/embedding, with explicit person-box association.

Only preprocessing, decoding, alignment and cosine decisions run on the host.
YuNet/SFace network requests go to GPU instances; no CPU model is loaded here.
"""
from __future__ import annotations

import cv2
import numpy as np

from demo.gpu.client import GpuPoseClient, require_gpu_config

TEMPLATE=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                   [41.5493,92.3655],[70.7299,92.2041]],np.float32)


def decode_yunet(outputs, image_size, scale, threshold=.8):
    width,height=image_size
    candidates=[]
    for stride in (8,16,32):
        n=(640//stride)**2
        values={key:np.asarray(outputs[f'{key}_{stride}']) for key in ('cls','obj','bbox','kps')}
        for key,channels in [('cls',1),('obj',1),('bbox',4),('kps',10)]:
            if values[key].shape!=(1,n,channels) or not np.isfinite(values[key]).all():
                raise ValueError('YuNet output shape or finite-value contract mismatch')
        score=np.sqrt(np.clip(values['cls'][0,:,0],0,1)*np.clip(values['obj'][0,:,0],0,1))
        indices=np.flatnonzero(score>=threshold)
        for index in indices:
            row,col=divmod(int(index),640//stride)
            bbox=values['bbox'][0,index]
            cx,cy=(bbox[0]+col)*stride,(bbox[1]+row)*stride
            bw,bh=np.exp(np.clip(bbox[2:],-20,20))*stride
            box=np.array([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2])/scale
            box[[0,2]]=box[[0,2]].clip(0,width)
            box[[1,3]]=box[[1,3]].clip(0,height)
            if box[2]<=box[0] or box[3]<=box[1]: continue
            points=values['kps'][0,index].reshape(5,2).copy()
            points[:,0]=(points[:,0]+col)*stride/scale
            points[:,1]=(points[:,1]+row)*stride/scale
            candidates.append({'box':box.tolist(),'score':float(score[index]),'landmarks':points.tolist()})
    if not candidates: return []
    xywh=[[f['box'][0],f['box'][1],f['box'][2]-f['box'][0],f['box'][3]-f['box'][1]] for f in candidates]
    keep=cv2.dnn.NMSBoxes(xywh,[f['score'] for f in candidates],threshold,.3)
    return [candidates[int(i)] for i in np.asarray(keep).reshape(-1)]


def align_face(image, points):
    points=np.asarray(points,np.float32)
    if points.shape!=(5,2) or not np.isfinite(points).all() or float(np.var(points,axis=0).sum())<1:
        raise ValueError('Five nondegenerate finite landmarks required')
    matrix=[]
    target=[]
    for (x,y),(u,v) in zip(points,TEMPLATE):
        matrix.extend([[x,-y,1,0],[y,x,0,1]])
        target.extend([u,v])
    a,b,tx,ty=np.linalg.lstsq(np.asarray(matrix),np.asarray(target),rcond=None)[0]
    transform=np.asarray([[a,-b,tx],[b,a,ty]],np.float32)
    return cv2.warpAffine(image,transform,(112,112),flags=cv2.INTER_LINEAR,borderValue=0)


class GpuIdentityClient(GpuPoseClient):
    def __init__(self,*args,body_model=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.body_model=body_model
        self._metadata={}

    def validate(self):
        pose=super().validate()
        self.identity_configs={}
        for name in ['yunet_gpu','sface_gpu',*([self.body_model] if self.body_model else [])]:
            config=self.json_get(f'/v2/models/{name}/config')
            require_gpu_config(config)
            self.identity_configs[name]=config
            self._metadata[name]=self.json_get(f'/v2/models/{name}')
        return pose

    def _network(self,name,array):
        metadata=self._metadata[name]
        return self.infer_arrays(name,{metadata['inputs'][0]['name']:array},
                                 [item['name'] for item in metadata['outputs']])

    def infer(self,bgr):
        result=super().infer(bgr)
        h,w=bgr.shape[:2]
        scale=min(640/w,640/h)
        rw,rh=round(w*scale),round(h*scale)
        canvas=np.zeros((640,640,3),np.uint8)
        canvas[:rh,:rw]=cv2.resize(bgr,(rw,rh))
        outputs=self._network('yunet_gpu',canvas.transpose(2,0,1)[None].astype(np.float32))
        faces=decode_yunet(outputs,(w,h),scale)
        result['faces']=faces
        used=set()
        for person in result['detections']:
            x1,y1,x2,y2=person['box']
            face=None
            for index,candidate in enumerate(faces):
                fx1,fy1,fx2,fy2=candidate['box']
                if index not in used and x1<=(fx1+fx2)/2<=x2 and y1<=(fy1+fy2)/2<=y2:
                    face=candidate
                    used.add(index)
                    break
            person['face']=face
            person['face_embedding']=None
            person['body_embedding']=None
            if face is not None:
                aligned=align_face(bgr,face['landmarks'])
                blob=aligned[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)
                vector=next(iter(self._network('sface_gpu',blob).values())).reshape(-1)
                person['face_embedding']=vector.tolist()
            if self.body_model:
                ix1,iy1,ix2,iy2=map(round,(max(0,x1),max(0,y1),min(w,x2),min(h,y2)))
                crop=bgr[iy1:iy2,ix1:ix2]
                if crop.size:
                    rgb=cv2.resize(crop,(128,256))[:,:,::-1].astype(np.float32)/255.
                    blob=((rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32))
                    vector=next(iter(self._network(self.body_model,blob.transpose(2,0,1)[None]).values())).reshape(-1)
                    person['body_embedding']=vector.tolist()
        return result
