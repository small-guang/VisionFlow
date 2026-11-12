# 第三方来源与许可

本文件说明当前 GPU 作品展示的发布边界，不将所有第三方内容重新许可为 Apache-2.0。

| 内容 | 来源与许可 | 本仓处理 |
| --- | --- | --- |
| 主仓自有工程代码 | 根 LICENSE，Apache-2.0 | 保留原声明；公开贡献范围以实际权利为准 |
| YOLO26m Pose 权重与上游实现 | Ultralytics，AGPL-3.0 | 仅选用其 pose 闭包；模型来源、摘要和完整 AGPL 声明位于推理仓 gpu_demo；权重与engine不收入Git；发布服务需保留适用许可和对应源码／导出信息 |
| YuNet / SFace | OpenCV Zoo固定版本：MIT / Apache-2.0 | 权重、目录许可证与声明快照在推理仓gpu_demo/licenses/identity |
| OSNet-AIN ReID | 作者kaiyangzhou/osnet固定权重仓库明确MIT | 保留模型卡、API、checkpoint LFS摘要与MIT声明；MSMT17训练数据适用独立条件，不声称训练数据开放 |
| NASA/JSC 真人访谈片段 | NASA 官方媒体；版权、人物与标志条件见链接 | 来源、裁剪与去音轨写入 demo/gpu/assets/source-manifest.json，不暗示机构或人物背书 |
| NumPy、OpenCV、Pillow、FFmpeg 等 | 各自许可证与第三方组件条款 | GPU 前后处理及媒体工具使用，不能把现成 OpenCV wheel 所含 IPP 当作全部开源 |
| Noto 中文字体 | SIL Open Font License | 用于流程示意图渲染，不打包系统字体 |
| Triton、CUDA、完整 TensorRT、NVIDIA 驱动 | 分别适用官方代码许可／软件条款 | GPU 运行依赖；不声称整个 GPU 软件栈全部开源 |

[素材详细说明](demo/gpu/MEDIA.md)、[当前模型制品清单](demo/gpu/identity-run-output/model-manifest.json)、[OpenCV wheel 第三方许可](https://github.com/opencv/opencv-python/blob/88/LICENSE-3RD-PARTY.txt)、[Ultralytics 许可](https://github.com/ultralytics/ultralytics/blob/v8.4.61/LICENSE)。

原有多模型工程保留 InsightFace、SBS、Face QA、CLIP 等适配代码，**不表示这些权重已获统一开源或再分发许可**；它们不进入当前 GPU 公开演示默认闭包。后续启用时应按推理仓来源清单逐项核验。
