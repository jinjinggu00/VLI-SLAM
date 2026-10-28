set -eo pipefail
base=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$base/upstream/ffreereg/tools/DepthPro/checkpoints" "$base/upstream/ffreereg/tools/YOHO/model/Backbone" "$base/upstream/ffreereg/tools/YOHO/model/PartI_train"
curl -fL --connect-timeout 15 --max-time 600 --retry 1 -o "$base/upstream/ffreereg/tools/DepthPro/checkpoints/depth_pro.pt" https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt
curl -fL --connect-timeout 15 --max-time 180 --retry 1 -o "$base/upstream/ffreereg/tools/YOHO/model/Backbone/best_val_checkpoint.pth" https://raw.githubusercontent.com/HpWang-whu/YOHO/master/model/Backbone/best_val_checkpoint.pth
curl -fL --connect-timeout 15 --max-time 180 --retry 1 -o "$base/upstream/ffreereg/tools/YOHO/model/PartI_train/model_best.pth" https://raw.githubusercontent.com/HpWang-whu/YOHO/master/model/PartI_train/model_best.pth
git clone --depth 1 --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git "$base/upstream/minkowskiengine"
ls -l /usr/local/cuda-12.1/bin/nvcc
