"""Header ordering compatibility only; original numeric source is unchanged."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
p=H/'upstream/minkowskiengine/src/spmm.cu'
before=p.read_text()
if '#include <torch/extension.h>' not in before or '#include <ATen/cuda/CUDAContext.h>' not in before:raise ValueError('Unexpected pinned author source')
after=before.replace('#include <torch/extension.h>\n','').replace('#include <ATen/cuda/CUDAContext.h>','#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>')
def numeric(s):return '\n'.join(x for x in s.splitlines() if not x.startswith('#include '))
if numeric(before)!=numeric(after):raise ValueError('Numeric code change forbidden')
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
(H/'minkowski_header_compatibility.json').write_text(json.dumps({'source_commit':'9f81ae66b33b883cd08ee4f64d08cf633608b118','file':str(p),'before_sha256':digest(before),'after_sha256':digest(after),'numeric_code_unchanged':True,'change':'Move existing original torch/extension.h before ATen CUDAUtils so at::Tensor has a complete type for PyTorch2.3.1. No solver/network numeric change.','basis':'Observed exact compiler error in native spmm.cu; CUDAUtils uses Tensor::get_device before complete definition.'},indent=2)+'\n')
p.write_bytes(after.encode())
print('Original numeric source unchanged; existing include order corrected.')
