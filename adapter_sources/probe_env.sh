set -eo pipefail
base=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$base/tmp"
"$base/feature_env/bin/python" - <<'PY'
import importlib.util,torch,sys,site
for name in ['kornia_rs','MinkowskiEngine','timm','einops','numpy','scipy']: print(name,importlib.util.find_spec(name))
print(sys.version);print(site.getsitepackages())
PY
command -v nvcc || true
ls -d /usr/local/cuda* || true
