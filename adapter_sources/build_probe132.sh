set -eo pipefail
base=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$base/tmp"
"$base/feature_env/bin/python" -c 'import torch;print(torch.__version__,torch.version.cuda)'
ls -l /usr/local/cuda-12.1/bin/nvcc || true
ls /usr/lib/x86_64-linux-gnu/libopenblas* || true
g++ --version | head -n1
