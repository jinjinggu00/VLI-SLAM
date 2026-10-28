import unittest,tempfile,hashlib
from pathlib import Path
from source_binding132 import verified_workspace_source
class AliasBinding(unittest.TestCase):
 def test_cross_surface_alias_still_binds_actual_bytes(self):
  root=Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent/'tmp')).resolve();f=root/'measured.f32';f.write_bytes(b'actual measured bytes')
  h=hashlib.sha256(f.read_bytes()).hexdigest();m={'/workspace/measured.f32':h}
  self.assertEqual(verified_workspace_source(f,root,m),h)
  f.write_bytes(b'changed')
  with self.assertRaises(ValueError):verified_workspace_source(f,root,m)
 def test_unknown_or_escaping_source_is_rejected(self):
  root=Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent/'tmp')).resolve();f=root/'x';f.write_bytes(b'x')
  with self.assertRaises(ValueError):verified_workspace_source(f,root,{})
  with self.assertRaises(ValueError):verified_workspace_source(root.parent/'external',root,{})
if __name__=='__main__':unittest.main(verbosity=2)
