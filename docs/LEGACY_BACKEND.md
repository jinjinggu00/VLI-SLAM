# 历史冻结观测后端运行说明

仅供自行保留了匹配封存输入的工程使用；本仓库不分发该输入包。这一入口不替代从原始bag开始的完整流水线。

在激光雷达与 IMU 信息基础上融合双目鱼眼图像，以 AirSLAM 为视觉惯性核心，加入图像回访、实测激光深度验证和轨迹—路标联合优化。

## 1. 准备目录

```bash
git clone https://github.com/jinjinggu00/VLI-SLAM.git
cd VLI-SLAM
```

此仓库提供比赛适配代码和最终联合图后端复现入口。原始 bag、冻结观测输入、权重和运行结果单独保留。准备已有的 `V132_reproduction.zip`，下面的操作不需要下载数据。

## 2. 恢复复现输入

用本地已有的复现包恢复18个输入文件，脚本逐项检查 SHA256，保留发生冲突的原文件并报错。

```bash
python3 tools/restore_inputs.py --archive /path/to/V132_reproduction.zip
```

恢复后，目录应包含：

```text
VLI-SLAM/
├── README.md                 # 操作步骤和运行命令
├── LICENSE                   # 项目许可
├── THIRD_PARTY_NOTICES.md     # 上游算法及许可说明
├── run_joint.sh              # 冻结观测后端复现入口
├── reproduce_joint.py        # 输入核对、原生编译及两场景复现
├── runtime.json              # 固定运行环境和文件哈希
├── src/                      # 原生C++联合优化与保留的上游因子
├── adapter_sources/          # 比赛数据适配、深度关联及图优化组织
│   └── native_gba/           # AirSLAM原生GBA相关源码
├── upstream/lightglue/       # 保留的LightGlue上游实现与许可
├── tools/restore_inputs.py   # 从已有复现包恢复输入
├── docs/CODE_STRUCTURE.md    # 代码结构和模块职责
├── licenses/                 # 第三方许可文件
└── inputs/                  # 本地恢复，Git忽略
    ├── scene_0001/
    └── scene_0002/
```

`inputs/` 已加入 `.gitignore`，不要将它提交到仓库。

## 3. 运行两场景后端复现

使用已保存的 AirSLAM Docker 环境。原记录中的镜像ID如下；该环境需要 Python 3、NumPy、SciPy、C++17、Eigen3、g2o 和 glog。

```bash
export SLAM_IMAGE='docker.io/xukuanhit/air_slam@sha256:9df819fdbb0ef0925adcd32f1f6d543e12272d4cc5c35118cb259597ebb508e7'
docker image inspect "$SLAM_IMAGE" >/dev/null

docker run --rm \
  --entrypoint /bin/bash \
  -v "$PWD:/reproduction" \
  -w /reproduction \
  "$SLAM_IMAGE" run_joint.sh
```

如果已经进入匹配的容器：

```bash
cd /reproduction
bash run_joint.sh
```

脚本先校验代码、输入和 g2o 库哈希，随后重新编译 `joint_bridge.cpp`，依次求解两个场景，并比较轨迹与三维路标。环境不匹配时先修复环境，不要删除哈希检查。

这一入口使用固定的真实观测，覆盖联合图重编译与求解；它的耗时不代表原始 bag、神经网络推理及完整前端的总耗时。

## 4. 查看输出

```bash
latest_run=$(find runs -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
python3 -m json.tool "$latest_run/report.json"
wc -l "$latest_run/scene_0001/public_trajectory.txt" \
      "$latest_run/scene_0002/public_trajectory.txt"
```

成功时终端输出 `STANDALONE TWO-SCENE JOINT REPRODUCTION PASSED`。两个轨迹文件应分别为5044行和4098行。每个场景还会保存 `poses.txt`、`poses.txt.landmarks.txt`、`optimizer.json` 和日志。

## 5. 在已有完整服务器工程中运行

以下命令用于已经恢复完整比赛 workspace、标定、上游代码、Python环境和模型的服务器。`adapter_sources/` 保留最终运行链路及其依赖，并沿用原工程的路径关系；不要把它当成只需 bag 即可运行的全新工程。

```bash
export WORKSPACE=/path/to/workspace
cd "$WORKSPACE/semantic_map/lidar_challenge_2026/submission_campaign_20261002/v132_author_aliked_crossmodal_20261009"

# 图像特征提取与匹配
bash features132.sh

# 激光深度关联与双向几何验证
feature_env/bin/python -B associate132.py
feature_env/bin/python -B verify132.py

# 最终联合图与导出
python3 -B graph132_r11.py
python3 -B package132_r11.py
python3 -B package132_r11.py audit
```

这些阶段要求对应输入和依赖已存在，且会拒绝覆盖部分已有输出。复跑时使用独立工作副本，保留原来的成功结果。原始双目校正、AirSLAM前端及原生GBA依赖原有完整工程，不包含在上面的独立后端入口内。

## 6. 代码位置与算法

| 位置 | 功能 |
| --- | --- |
| `adapter_sources/features132.py` | 公开预训练ALIKED-n16与LightGlue，回环匹配分支 |
| `adapter_sources/associate132.py` | LVI-SAM DepthRegister实测激光深度关联 |
| `adapter_sources/verify132.py` | 双向PnP、重投影及三维一致性验证 |
| `adapter_sources/gba_binding132.py` | 将原生AirSLAM GBA修正绑定到对应运动测量 |
| `adapter_sources/graph132_r11.py` | 最终运动约束、像素与深度观测组织 |
| `src/v132/joint_bridge.cpp` | 原生g2o观测边、鲁棒核和LM联合求解 |
| `adapter_sources/native_gba/` | AirSLAM原生GBA相关源码 |

AirSLAM主干保留PLNet点线提取、LightGlue点匹配与IMU处理；BoQ提供图像回访候选，ALIKED/LightGlue作为新增局部匹配分支，Point-LIO及点云几何提供辅助运动信息。团队实现集中在数据适配、多源绑定、验证、约束筛选和优化流程集成。仓库不包含未启用的实验候选、测试代码、运行日志或结果文件。

上游代码和各自许可保留在 `licenses/`、`upstream/` 与源码头部，详见 `THIRD_PARTY_NOTICES.md`。未重新训练上述公开网络。
