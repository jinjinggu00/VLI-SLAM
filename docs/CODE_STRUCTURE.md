# 代码结构与模块职责

## 可独立复现的入口

```text
run_joint.sh              # 后端复现入口，调用主程序
reproduce_joint.py        # 校验冻结输入与g2o库，编译并依次运行两场景
src/v132/joint_bridge.cpp # 调用g2o，联合求解机器人位姿和共享三维路标
runtime.json             # 原运行环境、源码与输入文件的SHA256记录
INPUTS_SHA256.json       # 从已有复现包恢复的18个输入文件清单
```

```bash
python3 tools/restore_inputs.py --archive /path/to/V132_reproduction.zip
bash run_joint.sh
```

第二条命令应在README所述的匹配环境中运行。该入口使用冻结的真实观测，不从原始bag重新运行图像前端或激光深度关联。

## 原比赛工程适配代码

```text
adapter_sources/
├── features132.py       # ALIKED-n16特征提取与LightGlue图像匹配
├── associate132.py      # 关联实测激光深度、采集时间和图像特征编号
├── verify132.py         # 双向PnP、重投影及三维几何一致性验证
├── gba_binding132.py    # 将AirSLAM原生GBA修正绑定到对应运动测量
├── graph132_r11.py      # 最终联合图的位姿、路标、运动及像素深度观测组织
├── package132_r11.py    # 最终轨迹导出和公开格式检查
├── bridge_io132.py      # 观测与文件读写接口
└── native_gba/          # AirSLAM原生GBA相关源码
```

适配代码仅保留最终运行链路与必要依赖，保持原工程相对路径。最终入口以`graph132_r11.py`、`package132_r11.py`及固定复现入口为准。`adapter_sources`不能脱离完整服务器工程直接作为原始bag的一键入口。

## 模型与算法分工

- AirSLAM：PLNet点线提取、LightGlue点匹配及视觉惯性估计。权重沿用公开模型。
- Point-LIO与点云配准：提供激光惯性和几何运动辅助信息。
- BoQ：提出图像回访候选。
- ALIKED-n16与LightGlue：独立补充图像特征对应分支。
- DepthRegister：关联实测激光深度。
- Kabsch与双向PnP：检查三维与投影几何一致性。
- AirSLAM内部GBA与新增g2o联合图：处理视觉惯性地图及机器人位姿、共享路标的联合优化。

团队工作集中在数据适配、观测身份与时间绑定、跨模态深度关联、几何验证和优化流程集成。未把公开网络或求解器声明为自研模型。

## 本地材料

原始bag、模型权重、冻结观测输入、历史运行结果以及答辩和个人材料不放入Git仓库。按README使用已有本地文件，不在README提供数据下载链接。

## 必要依赖的保留原则

原生联合优化源码还包含一个保留原路径的上游重力因子头文件。该目录名虽然带历史标识，但当前C++文件确实引用它，不能按目录名称删除。运行清单保留18个本地输入的哈希，输入内容本身不上传。已有服务器工程适配代码仍依赖完整工程的标定、数据、模型及前置阶段输入，不能据此将独立后端称为原始bag全前端复现。
