import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from demo.gpu.finalize_video import matching_source_manifest


class FinalizeTests(unittest.TestCase):
    def test_other_video_does_not_inherit_nasa_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.json'
            source.write_text(json.dumps({'output':{'sha256':'official-source'}}))
            self.assertIsNone(matching_source_manifest(source, 'my-different-video'))
            self.assertEqual(matching_source_manifest(source, 'official-source')['output']['sha256'],
                             'official-source')

    def test_server_evidence_failure_preserves_existing_artifact_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            (directory/'manifest.json').write_text(json.dumps({'output':{'frames':1}}))
            (directory/'annotated.mp4').write_bytes(b'original-video-payload')
            original=(directory/'manifest.json').read_bytes()
            result=subprocess.run([sys.executable,'-B','demo/gpu/finalize_video.py',
                '--output',str(directory),'--container','vf-no-such-container'],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual((directory/'annotated.mp4').read_bytes(),b'original-video-payload')
            self.assertEqual((directory/'manifest.json').read_bytes(),original)
