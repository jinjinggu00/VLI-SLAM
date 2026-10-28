set -eo pipefail
base=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=2 CUDA_HOME=/usr/local/cuda-12.1 CUDA_PATH=/usr/local/cuda-12.1
export PATH="$base/feature_env/bin:$CUDA_HOME/bin:$PATH" TMPDIR="$base/tmp" PIP_CACHE_DIR="$base/tmp/pip_cache" TORCH_CUDA_ARCH_LIST=8.9
export NVCC_PREPEND_FLAGS="-include thrust/execution_policy.h -include thrust/unique.h -include thrust/remove.h -include thrust/reduce.h -include thrust/sort.h"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py="$base/feature_env/bin/python"
blas="$base/upstream/blas_runtime/usr/lib/x86_64-linux-gnu/openblas-pthread"
export LIBRARY_PATH="$blas:$LIBRARY_PATH" LD_LIBRARY_PATH="$blas:$LD_LIBRARY_PATH"
cd "$base/upstream/minkowskiengine"
git rev-parse HEAD
timeout 1800 "$py" setup.py install --force_cuda --blas=openblas --blas_include_dirs="$base/upstream/blas_runtime/usr/include/x86_64-linux-gnu/openblas-pthread" --blas_library_dirs="$blas"
cd "$base"
"$py" -m pip install pillow_heif==0.18.0 --no-deps --timeout 20 --retries 1
"$py" -c 'import MinkowskiEngine,torch;print(MinkowskiEngine.__version__,torch.__version__)'
