"""Physical file hash binding across read-only /workspace and WSL aliases."""
from pathlib import Path
import hashlib
def verified_workspace_source(path,workspace,manifest):
 path=Path(path).resolve();root=Path(workspace).resolve()
 relative=path.relative_to(root)
 key='/workspace/'+relative.as_posix()
 expected=manifest.get(key)
 if expected is None or not path.is_file():raise ValueError('Missing original source identity')
 actual=hashlib.sha256(path.read_bytes()).hexdigest()
 if actual!=expected:raise ValueError('Changed actual protected source bytes')
 return actual
