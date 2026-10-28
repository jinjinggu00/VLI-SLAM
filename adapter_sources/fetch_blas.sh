set -eo pipefail
base=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$base/upstream/blas_debs" "$base/upstream/blas_runtime"
cd "$base/upstream/blas_debs"
apt-get download libopenblas-pthread-dev libopenblas0-pthread
for f in ./*.deb; do dpkg-deb -x "$f" "$base/upstream/blas_runtime"; done
ls -l "$base/upstream/blas_runtime/usr/include/x86_64-linux-gnu/openblas-pthread/cblas.h"
ls -l "$base/upstream/blas_runtime/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so"
