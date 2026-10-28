from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from hashlib import sha256
import argparse,json

def main():
    p=argparse.ArgumentParser(description='Restore only the eighteen measured input files from an existing local reproduction archive.')
    p.add_argument('--archive',required=True,type=Path)
    p.add_argument('--dest',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();root=a.dest.resolve();root.mkdir(parents=True,exist_ok=True)
    expected=json.loads((Path(__file__).resolve().parents[1]/'INPUTS_SHA256.json').read_text())
    with ZipFile(a.archive) as z:
        pending=[]
        for name,digest in expected.items():
            q=PurePosixPath(name)
            if q.is_absolute() or '..' in q.parts or not name.startswith('inputs/'):raise ValueError('Unsafe input name')
            target=root.joinpath(*q.parts)
            if not target.resolve().is_relative_to(root):raise ValueError('Input escapes destination')
            if any(p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction()) for p in [target,*list(target.parents)[:len(q.parts)-1]]):raise ValueError('Link or junction input target')
            data=z.read(name)
            if sha256(data).hexdigest()!=digest:raise ValueError('Input checksum differs: '+name)
            if target.exists():
                if not target.is_file() or sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Keep conflicting existing file: '+name)
            else:pending.append((target,data))
        for target,data in pending:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
    print('INPUTS READY: 18 checked files; newly restored:',len(pending))
if __name__=='__main__':main()
