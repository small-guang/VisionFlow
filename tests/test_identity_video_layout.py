import unittest
import numpy as np
from demo.gpu.run import compose_frame


class IdentityVideoLayoutTests(unittest.TestCase):
    def test_identity_gallery_does_not_change_source_or_annotated_panel_width(self):
        frame=np.full((360,640,3),50,np.uint8)
        rendered=compose_frame(frame,{'identity_enabled':True,'detections':[]},0,30,{})
        self.assertEqual(rendered.shape,(460,1520,3))
        np.testing.assert_array_equal(rendered[55:415,:640],frame)
        np.testing.assert_array_equal(rendered[55:415,640:1280],frame)
