set -e
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tmp"
exec "$PWD/feature_env/bin/python" -B upstream_manifest132.py
