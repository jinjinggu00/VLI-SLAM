set -eo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/tmp" TORCH_HOME="$PWD/tmp/torch_cache" CUDA_CACHE_PATH="$PWD/tmp/cuda_cache" CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PWD/feature_env/bin/python" -B features132.py "$@"
