# 公开依赖与环境

## 作者环境

公开镜像：`docker.io/xukuanhit/air_slam:v4`。

固定运行引用：

```text
docker.io/xukuanhit/air_slam@sha256:9df819fdbb0ef0925adcd32f1f6d543e12272d4cc5c35118cb259597ebb508e7
```

公开Docker Hub清单为linux/amd64、16层，压缩层合计18,356,384,900字节。摘要固定的是镜像清单；不同Docker存储后端显示的本地image ID不一定采用同一含义。使用`docker pull 仓库名@sha256:摘要`，不要把一个不存在于本机的裸SHA当作镜像获取方式。

作者原生依赖包括ROS Noetic、OpenCV 4.2、Eigen3、CUDA 12.1、TensorRT 8.6.1.6、g2o、glog/gflags、yaml-cpp、Boost。作者README还列出Ceres 2.0.0。模型推理与GPU环境需另外验证；容器能启动不证明整个项目已经构建成功。

- [AirSLAM固定版本及作者安装说明](https://github.com/sair-lab/AirSLAM/tree/d1265771026091db97ee8f7a6d80692f0edd5fef)
- [Docker Hub镜像条目](https://hub.docker.com/r/xukuanhit/air_slam)

## 源码与模型校验

`config/dependencies.json`记录源码commit、公开模型URL、字节数和SHA256。`tools/bootstrap.py`仅使用Python标准库；不需要先安装Torch或恢复场景文件。

| 内容 | 固定版本/来源 |
| --- | --- |
| AirSLAM | `d1265771026091db97ee8f7a6d80692f0edd5fef` |
| LightGlue | `eb42fee2d71449efb0aa5c10549752b5d75384d8` |
| PLNet S0/S1、SuperPoint、SuperPoint LightGlue ONNX | AirSLAM上述commit的`output/` |
| ALIKED-n16 | 作者公开检查点，必须匹配配置中的完整SHA256 |
| ALIKED LightGlue | cvg/LightGlue release `v0.1_arxiv` |
| BoQ ResNet50 | Bag-of-Queries release `v1.0` |
| ResNet50 backbone | PyTorch官方检查点 |

模型下载只写入指定本地目录，校验失败不接受文件；已存在的不同文件不会被覆盖。源地址即使发生变更，也必须继续匹配原校验值。网络失败时恢复正常HTTPS访问后重试，不关闭证书验证，也不要删除SHA校验。

`.onnx`为模型计算图；`.engine`绑定TensorRT/GPU环境，不能用作者ONNX的SHA为旧引擎缓存背书。原始作者S1与历史no-scatter兼容图不是同一文件；配置只指向原始作者S1，任何兼容转换需要单独公布转换源码与验证结果。

## Python特征运行环境的边界

原AirSLAM Noetic环境与新增图像匹配分支分开管理。历史特征环境记录了Python3.10、PyTorch2.3.1/cu121等版本，但旧安装脚本借用了其他实验环境的site-packages；它不是面向干净机器的完整安装配方。不得通过把历史环境目录加入PYTHONPATH来宣称独立复现。

当前引导工具不安装或借用这些私有环境。完整特征推理、BoQ、Point-LIO及LiDAR约束构建仍需在原始流水线源码闭包确认后，分别锁定并验证实际依赖、构建顺序和运行接口。
