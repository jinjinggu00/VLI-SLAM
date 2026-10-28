set -eo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/tmp" TORCH_HOME="$PWD/tmp/torch_cache" CUDA_CACHE_PATH="$PWD/tmp/cuda_cache" CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH="$PWD/upstream/fortran_runtime/usr/lib/x86_64-linux-gnu:$PWD/upstream/blas_runtime/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH"
exec timeout 1200 "$PWD/feature_env/bin/python" -B crossmodal132.py
