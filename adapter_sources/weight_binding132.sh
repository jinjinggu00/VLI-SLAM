set -eo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_HOME="$PWD/tmp/torch_cache" TMPDIR="$PWD/tmp"
exec "$PWD/feature_env/bin/python" -B weight_binding132.py
