set -e
base=$(cd -- "$(dirname -- "$0")" && pwd)
git clone --depth 1 https://github.com/cvg/LightGlue.git "$base/upstream/lightglue"
git clone --depth 1 --branch FFreeReg https://github.com/WHU-USI3DV/FreeReg.git "$base/upstream/ffreereg"
git -C "$base/upstream/lightglue" rev-parse HEAD
git -C "$base/upstream/ffreereg" rev-parse HEAD
