"""Face-anchored identity decisions for a small, anonymous demonstration gallery.

Body similarity only proposes candidates. It never independently confirms a
person or contaminates another identity's prototypes. Model spaces are fixed
for the lifetime of this gallery and must not be mixed between runs.
"""
from __future__ import annotations

import numpy as np


class IdentityGallery:
    def __init__(self, face_threshold=.5, body_threshold=.8):
        if not -1 <= face_threshold <= 1 or not -1 <= body_threshold <= 1:
            raise ValueError('Cosine thresholds must be within [-1,1]')
        self.face_threshold=face_threshold
        self.body_threshold=body_threshold
        self._people=[]
        self._dimensions={}

    @property
    def size(self):
        return len(self._people)

    def _vector(self, value, space):
        if value is None: return None
        vector=np.asarray(value,dtype=np.float32)
        if vector.ndim!=1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError('Embedding must be a finite nonempty vector')
        norm=float(np.linalg.norm(vector))
        if norm < 1e-9: raise ValueError('Embedding norm must be positive')
        dimension=self._dimensions.setdefault(space,vector.size)
        if dimension != vector.size: raise ValueError('Embedding space dimension changed')
        return vector/norm

    def _best(self, vector, space):
        matches=[]
        if vector is not None:
            for person in self._people:
                prototypes=person[space]
                if prototypes:
                    matches.append((max(float(vector @ proto) for proto in prototypes), person))
        return max(matches,key=lambda item:item[0]) if matches else (None,None)

    def observe(self, *, face_embedding=None, body_embedding=None):
        face=self._vector(face_embedding,'face')
        body=self._vector(body_embedding,'body')
        body_score,body_person=self._best(body,'body')
        body_candidate=body_person if body_score is not None and body_score>=self.body_threshold else None
        result={'identity_id':None,'candidate_id':body_candidate['id'] if body_candidate else None,
                'status':'body_candidate' if body_candidate else 'unconfirmed',
                'face_similarity':None,'body_similarity':body_score,'body_conflict':False}
        if face is None: return result
        face_score,person=self._best(face,'face')
        result['face_similarity']=face_score
        if person is None or face_score<self.face_threshold:
            person={'id':f'Demo-P{len(self._people)+1:03d}','face':[],'body':[]}
            self._people.append(person)
            result['status']='enrolled_face'
        else:
            result['status']='face_confirmed'
        result['identity_id']=person['id']
        result['body_conflict']=body_candidate is not None and body_candidate['id']!=person['id']
        # Keep a small, face-confirmed prototype set. No update from body-only
        # observations; never alter a body candidate rejected by the face.
        if len(person['face'])<5: person['face'].append(face.copy())
        if body is not None and len(person['body'])<5: person['body'].append(body.copy())
        return result
