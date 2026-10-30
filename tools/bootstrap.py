"""Acquire public dependencies; never distribute or restore competition data.

Python 3.8+ standard library only. This prepares dependencies, not a complete
raw-bag SLAM run. Model bytes and sizes are verified before accepting a file.
"""
import argparse
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config' / 'dependencies.json'
ALLOWED_HOSTS = {'github.com', 'raw.githubusercontent.com', 'download.pytorch.org'}


def https_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Only configured public HTTPS upstream URLs are supported')
    return value


def validate_model(item):
    name = item.get('name', '')
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or '/' in name or '\\' in name or ':' in name
            or ntpath.isabs(name) or Path(name).suffix not in ('.onnx', '.pth')):
        raise ValueError('Model name must be a single ONNX/PTH file name')
    if not re.fullmatch('[0-9a-f]{64}', item.get('sha256', '')):
        raise ValueError('A complete SHA-256 is required for every model')
    if type(item.get('bytes')) is not int or item['bytes'] <= 0:
        raise ValueError('A positive model size is required')
    if item.get('group') not in ('airslam', 'matching', 'retrieval'):
        raise ValueError('Unknown model group')
    https_url(item['url'])


def load_config():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported dependency configuration')
    if not re.fullmatch(r'docker\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}', config['image']):
        raise ValueError('A public repository and immutable image digest are required')
    if config['platform'] != 'linux/amd64':
        raise ValueError('The author image supports linux/amd64')
    names = set()
    for item in config['models']:
        validate_model(item)
        if item['name'] in names:
            raise ValueError('Duplicate model target')
        names.add(item['name'])
    for source in config['sources'].values():
        https_url(source['url'])
        if not re.fullmatch('[0-9a-f]{40}', source['commit']):
            raise ValueError('Every source checkout must use a full commit ID')
    return config


def plain_path(path):
    """Reject symlink/junction output paths instead of writing through them."""
    path = Path(path).expanduser().absolute()
    for candidate in (path, *path.parents):
        reparse = False
        if os.name == 'nt':
            try:
                attributes = candidate.lstat().st_file_attributes
                reparse = bool(attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400))
            except FileNotFoundError:
                pass
        if candidate.is_symlink() or reparse:
            raise ValueError('Output must not pass through a symbolic link or junction')
    return path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def matching_file(path, item):
    return path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256']


def ensure_model(destination, item, download=False, opener=urlopen):
    validate_model(item)
    destination = plain_path(destination)
    target = plain_path(destination / item['name'])
    result = dict(name=item['name'], path=str(target), sha256=item['sha256'])
    if target.exists():
        if not matching_file(target, item):
            raise ValueError('Existing model differs and was preserved: ' + str(target))
        return dict(result, state='verified')
    if not download:
        return dict(result, state='missing')
    destination.mkdir(parents=True, exist_ok=True)
    plain_path(destination)
    fd, temporary = tempfile.mkstemp(prefix='.' + item['name'] + '.', suffix='.part', dir=str(destination))
    temporary = Path(temporary)
    try:
        checksum = hashlib.sha256()
        total = 0
        request = Request(item['url'], headers={'User-Agent': 'VLI-SLAM-dependency-bootstrap/1'})
        with os.fdopen(fd, 'wb') as output, opener(request, timeout=45) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                total += len(chunk)
                if total > item['bytes']:
                    raise ValueError('Download exceeds the pinned model size')
                output.write(chunk)
                checksum.update(chunk)
        if total != item['bytes'] or checksum.hexdigest() != item['sha256']:
            raise ValueError('Downloaded model size/SHA-256 mismatch: ' + item['name'])
        plain_path(target)
        try:
            # Atomic no-overwrite promotion on the same filesystem.
            os.link(str(temporary), str(target))
        except FileExistsError:
            if not matching_file(target, item):
                raise ValueError('A concurrent conflicting model was preserved: ' + str(target))
        return dict(result, state='downloaded', bytes=total)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_run(folder, arguments, capture=False):
    environment = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(['git', '-C', str(folder)] + arguments,
                            check=True, text=True, stdout=subprocess.PIPE if capture else None,
                            env=environment, timeout=180)
    return result.stdout.strip() if capture else None


def ensure_source(destination, name, config, fetch=False):
    source = config['sources'][name]
    destination = plain_path(destination)
    result = dict(name=name, path=str(destination), commit=source['commit'])
    if destination.exists():
        if not (destination / '.git').is_dir():
            raise ValueError('Existing destination is not an owned source checkout; preserved')
        actual = git_run(destination, ['rev-parse', 'HEAD'], capture=True)
        if actual != source['commit']:
            raise ValueError('Existing source commit differs; no reset or overwrite was performed')
        if git_run(destination, ['status', '--porcelain'], capture=True):
            raise ValueError('Existing upstream source has local changes; preserved')
        return dict(result, state='verified')
    if not fetch:
        return dict(result, state='missing')
    if not shutil.which('git'):
        raise ValueError('git is required for source acquisition')
    destination.parent.mkdir(parents=True, exist_ok=True)
    plain_path(destination.parent)
    temporary = Path(tempfile.mkdtemp(prefix='.' + name + '-source-', dir=str(destination.parent)))
    git_run(temporary, ['init', '--quiet'])
    git_run(temporary, ['remote', 'add', 'origin', source['url']])
    git_run(temporary, ['fetch', '--depth=1', 'origin', source['commit']])
    git_run(temporary, ['checkout', '--detach', 'FETCH_HEAD'])
    if git_run(temporary, ['rev-parse', 'HEAD'], capture=True) != source['commit']:
        raise ValueError('Downloaded upstream revision differs; temporary checkout preserved')
    if destination.exists():
        raise ValueError('Source destination was concurrently created; temporary checkout preserved')
    temporary.rename(destination)
    return dict(result, state='downloaded')


def check_bag(path):
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('A regular ROS1 bag file is required')
    with path.open('rb') as stream:
        header = stream.read(13)
    if header != b'#ROSBAG V2.0\n':
        raise ValueError('Not a ROS1 bag V2 file: ' + str(path))
    return dict(path=str(path), bytes=path.stat().st_size, rosbag_v2_header=True,
                sensor_content_verified=False, slam_run_verified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    image = commands.add_parser('image', help='Print the public pinned Docker pull command')
    image.add_argument('--pull', action='store_true', help='Actually pull the roughly 18.36 GB compressed image')
    models = commands.add_parser('models', help='List, verify or download original pretrained weights')
    models.add_argument('--group', choices=['all', 'airslam', 'matching', 'retrieval'], default='all')
    models.add_argument('--destination', type=Path, default=ROOT / '.deps' / 'models')
    models.add_argument('--download', action='store_true', help='Download missing files and verify them')
    models.add_argument('--list', action='store_true', help='List public acquisition metadata without touching files')
    source = commands.add_parser('source', help='Verify or fetch a pinned upstream source checkout')
    source.add_argument('--name', choices=['airslam', 'lightglue'], required=True)
    source.add_argument('--destination', type=Path, required=True)
    source.add_argument('--fetch', action='store_true')
    bags = commands.add_parser('bags', help='Read-only ROS1 bag header precheck; does not verify sensor topics')
    bags.add_argument('--scene-1', type=Path, required=True)
    bags.add_argument('--scene-2', type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config()
    if args.action == 'image':
        command = ['docker', 'pull', '--platform', config['platform'], config['image']]
        if args.pull:
            if not shutil.which('docker'):
                raise ValueError('Docker CLI is not installed or not available in PATH')
            subprocess.run(command, check=True)
        print(json.dumps(dict(command=command, executed=args.pull,
                              compressed_bytes=config['image_compressed_bytes'],
                              scope='Author environment acquisition only; no SLAM or source build was run'), indent=2))
        return 0
    if args.action == 'models':
        selected = [x for x in config['models'] if args.group == 'all' or x['group'] == args.group]
        if args.list:
            print(json.dumps(selected, indent=2))
            return 0
        reports = []
        for item in selected:
            print('CHECK ' + item['name'], file=sys.stderr, flush=True)
            reports.append(ensure_model(args.destination, item, args.download))
        print(json.dumps(dict(models=reports, model_payloads_are_not_part_of_repository=True,
                              slam_run_verified=False), indent=2))
        return 2 if any(x['state'] == 'missing' for x in reports) else 0
    if args.action == 'source':
        report = ensure_source(args.destination, args.name, config, args.fetch)
        print(json.dumps(report, indent=2))
        return 2 if report['state'] == 'missing' else 0
    reports = [check_bag(args.scene_1), check_bag(args.scene_2)]
    if reports[0]['path'] == reports[1]['path']:
        raise ValueError('Two distinct scene bag files are required')
    print(json.dumps(dict(bags=reports, scope='File/header precheck only; no decoding, topic/time/calibration check or SLAM run'), indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print('DEPENDENCY PREPARATION FAILED: ' + str(error), file=sys.stderr)
        sys.exit(1)
