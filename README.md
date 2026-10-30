# VLI-SLAM

视觉、激光与惯性融合 SLAM。保留 AirSLAM 点线视觉惯性主干，结合图像回访、实测激光深度验证和位姿—三维地图点联合优化。

## 运行范围

本仓库只提供源码、配置与公开依赖获取工具。使用者自行准备官方原始 ROS1 bag；不分发数据集、校正图像、场景缓存、冻结观测、参考轨迹或结果压缩包。

当前可独立执行的是公开环境、上游源码和预训练模型的获取与校验。**从原始 bag 开始的完整比赛流水线尚未完成独立发布与验证。** `adapter_sources/` 保留最终比赛适配实现，仍有历史工作目录依赖；不能把下列准备命令视为已经跑通全流程或得到官方成绩。

## 1. 获取项目

在 Linux 或 WSL 终端运行，准备 Python 3.8+、Git、Docker；GPU 推理还需要 NVIDIA 驱动与容器 GPU 支持。

```bash
git clone https://github.com/jinjinggu00/VLI-SLAM.git
cd VLI-SLAM
python3 tools/bootstrap.py --help
```

## 2. 获取作者 Docker 环境

镜像使用公开仓库名和不可变摘要，不要求本机预先保存镜像。下载的压缩层约18.36 GB，还需要解压与构建空间。

```bash
python3 tools/bootstrap.py image
python3 tools/bootstrap.py image --pull

export SLAM_IMAGE='docker.io/xukuanhit/air_slam@sha256:9df819fdbb0ef0925adcd32f1f6d543e12272d4cc5c35118cb259597ebb508e7'
docker image inspect "$SLAM_IMAGE" >/dev/null
```

`image`只显示拉取命令；加`--pull`才执行下载。作者环境为Ubuntu 20.04 / ROS Noetic / CUDA 12.1 / TensorRT 8.6，具体来源见[依赖说明](docs/DEPENDENCIES.md)。镜像获取不等于项目已经编译或SLAM已经运行。

需要进入容器检查环境时：

```bash
docker run --rm -it --gpus all \
  --entrypoint /bin/bash \
  --mount "type=bind,source=$PWD,target=/reproduction" \
  -w /reproduction "$SLAM_IMAGE"

# 容器内
source /opt/ros/noetic/setup.bash
python3 --version
nvidia-smi
```

## 3. 获取和核对公开预训练模型

模型直接来自作者公开地址；每个文件必须同时通过大小和SHA256校验。下载文件默认放在被Git忽略的`.deps/models/`，不上传到本仓库。

```bash
# 查看来源与校验值，不下载
python3 tools/bootstrap.py models --list

# 下载缺失文件；已有且不一致的文件会被保留并报错
python3 tools/bootstrap.py models --group airslam --download
python3 tools/bootstrap.py models --group matching --download
python3 tools/bootstrap.py models --group retrieval --download

# 只检查，缺少模型时返回退出码2，不创建文件
python3 tools/bootstrap.py models
```

三个分组分别包含AirSLAM的4个ONNX模型、ALIKED/LightGlue的2个检查点、BoQ相关的2个检查点。实际TensorRT引擎需在目标GPU与匹配环境中重新生成；不提供旧`.engine`缓存。

## 4. 获取固定版本上游源码

```bash
python3 tools/bootstrap.py source --name airslam \
  --destination "$PWD/.deps/AirSLAM" --fetch
python3 tools/bootstrap.py source --name lightglue \
  --destination "$PWD/.deps/LightGlue" --fetch
```

脚本使用完整commit ID，拒绝覆盖不同版本或存在本地修改的目录。这一步只获取上游源码，不会构建或运行比赛算法；其他上游依赖及项目适配仍见[当前接口与范围](docs/REPRODUCTION_SCOPE.md)。

## 5. 准备自己的原始输入

原始数据存放在仓库外。以下示例中的路径由使用者替换；本仓库不提供数据下载链接或数据文件。

```bash
export BAG_ROOT='/absolute/path/to/official_bags'
python3 tools/bootstrap.py bags \
  --scene-1 "$BAG_ROOT/sequence_01_0000_0550.bag" \
  --scene-2 "$BAG_ROOT/sequence_02_0560_0970.bag"
```

此命令只读检查两份文件的ROS1 bag V2头，不解码、修改或上传输入。它不校验传感器topic、时间关联或标定，也不代表数据已通过完整SLAM检查。

## 算法与代码

| 位置 | 作用 |
| --- | --- |
| `tools/bootstrap.py`、`config/dependencies.json` | 公开环境、固定源码与检查点获取，文件级校验 |
| `adapter_sources/features132.py` | ALIKED-n16与LightGlue图像特征匹配分支 |
| `adapter_sources/associate132.py` | LVI-SAM DepthRegister实测激光深度关联 |
| `adapter_sources/verify132.py` | 双向PnP、重投影及实测三维一致性验证 |
| `adapter_sources/gba_binding132.py` | 将原生AirSLAM GBA修正绑定到运动测量 |
| `adapter_sources/graph132_r11.py` | 组织运动、像素与深度联合观测 |
| `src/v132/joint_bridge.cpp` | g2o原生观测边、鲁棒核与联合求解 |

AirSLAM保留PLNet点线特征、LightGlue点匹配及IMU处理；BoQ提供回访候选；Point-LIO和点云几何提供辅助运动约束。团队实现集中在多源数据适配、坐标与时间绑定、几何验证、约束筛选和优化流程集成。模型使用公开预训练权重，没有重新训练这些网络。

## 已有封存输入的后端复现

`run_joint.sh`和`reproduce_joint.py`是历史最终联合图后端入口，只适用于自行保留了匹配冻结观测的工程。它们不包含原始bag解码或前端推理，也不是本仓库向新用户提供的从头复现入口。该接口的说明保留在[历史后端运行说明](docs/LEGACY_BACKEND.md)；本仓库不分发其输入包。

上游来源与许可见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。运行中生成的数据、模型、构建文件、日志与结果由`.gitignore`排除。
