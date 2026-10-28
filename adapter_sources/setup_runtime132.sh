set -eo pipefail
base=$(cd -- "$(dirname -- "$0")" && pwd)
old=$(dirname "$base")/v131_author_gba_appearance_lvi_20261009/unicorrn_env
site=$("$base/feature_env/bin/python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$old/lib/python3.10/site-packages" > "$site/prior_readonly_packages.pth"
export PIP_CACHE_DIR="$base/tmp/pip_cache" TMPDIR="$base/tmp"
"$base/feature_env/bin/python" -m pip install kornia==0.7.3 kornia_rs==0.1.9 timm==0.9.5 --no-deps --timeout 20 --retries 1
"$base/feature_env/bin/python" -c 'import torch,torchvision,kornia,open3d,cv2;print(torch.__version__,torchvision.__version__,kornia.__version__);print(torch.cuda.is_available())'
