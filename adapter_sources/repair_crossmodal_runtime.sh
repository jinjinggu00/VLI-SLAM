set -eo pipefail
cd -- "$(dirname -- "$0")"
base="$PWD"
mkdir -p "$base/tmp/fortran_debs" "$base/upstream/fortran_runtime"
cd "$base/tmp/fortran_debs"
apt-get download libgfortran5 libquadmath0
for deb in "$PWD"/*.deb; do dpkg-deb -x "$deb" "$base/upstream/fortran_runtime"; done
cd "$base"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$base/upstream/fortran_runtime/usr/lib/x86_64-linux-gnu:$base/upstream/blas_runtime/usr/lib/x86_64-linux-gnu/openblas-pthread:$LD_LIBRARY_PATH"
"$base/feature_env/bin/python" -B -c 'import runtime_compat132;import MinkowskiEngine,torch,numpy;print("DIRECT RUNTIME IMPORT",MinkowskiEngine.__version__,torch.__version__,numpy.__version__)'
