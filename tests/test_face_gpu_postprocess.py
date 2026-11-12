import unittest
import numpy as np
from demo.gpu.identity_client import decode_yunet, align_face


class FacePostprocessTests(unittest.TestCase):
    def test_yunet_grid_decode_restores_boxes_and_landmarks(self):
        arrays={}
        for stride in (8,16,32):
            n=(640//stride)**2
            for key,channels in [('cls',1),('obj',1),('bbox',4),('kps',10)]:
                arrays[f'{key}_{stride}']=np.zeros((1,n,channels),np.float32)
        index=82  # y=1, x=2 in the stride-8 grid.
        arrays['cls_8'][0,index,0]=.81
        arrays['obj_8'][0,index,0]=.81
        faces=decode_yunet(arrays,(640,640),1.)
        self.assertEqual(len(faces),1)
        np.testing.assert_allclose(faces[0]['box'],[12,4,20,12])
        np.testing.assert_allclose(faces[0]['landmarks'],[[16,8]]*5)

    def test_nonfinite_network_outputs_are_rejected(self):
        arrays={}
        for stride in (8,16,32):
            n=(640//stride)**2
            for key,channels in [('cls',1),('obj',1),('bbox',4),('kps',10)]:
                arrays[f'{key}_{stride}']=np.zeros((1,n,channels),np.float32)
        arrays['cls_8'][0,0,0]=np.nan
        with self.assertRaises(ValueError): decode_yunet(arrays,(640,640),1.)

    def test_five_landmark_alignment_preserves_template_and_rejects_degenerate(self):
        points=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],
                         [41.5493,92.3655],[70.7299,92.2041]],np.float32)
        image=np.zeros((112,112,3),np.uint8)
        image[50:60,50:60]=[12,34,56]
        result=align_face(image,points)
        self.assertEqual(result.shape,(112,112,3))
        np.testing.assert_array_equal(result[55,55],image[55,55])
        with self.assertRaises(ValueError): align_face(image,np.zeros((5,2),np.float32))
