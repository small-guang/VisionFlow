# GPU 人脸与人形身份确认演示

[展示页](index.html) · [实际视频](identity-run-output/annotated.mp4) · [逐帧结果](identity-run-output/events.jsonl) · [运行证据](identity-run-output/manifest.json) · [模型清单](identity-run-output/model-manifest.json) · [素材许可](MEDIA.md)

真实动态访谈按 A 5秒 → B 5秒 → A 5秒剪辑，画面保留人工剪辑标识。YOLO26m pose 提供人形与姿态，YuNet 检测人脸及五点，SFace 特征完成匿名登记与确认，再将身份空间关联到人形；OSNet-AIN 专用人形特征只提供候选。右侧图库由本次实际登记帧形成，不是预置人物库。

本例实际得到 Demo-P001 → Demo-P002 → Demo-P001，图库2人。225个输出帧，YuNet/SFace各225次、OSNet229次；B段4个额外无脸人形检测只保留候选、未确认或新增身份。人脸阈值0.5、人形候选阈值0.8仅是演示配置，不代表通用准确率或生产阈值。源码不读取 Source A/B 字幕决定身份。

## 1. 准备环境与 GPU 服务

两个仓库以 `VisionFlow/`、`visionflow-inference/` 同级放置。需要 NVIDIA GPU、匹配的驱动与容器运行环境、Docker Compose。客户端及素材工具建议使用 Python 3.12 独立环境：

```powershell
# 在 VisionFlow 根目录
python -m pip install -r demo/gpu/requirements.txt
```

在推理仓按其 `gpu_demo/README.md` 安装校验依赖，准备已授权且摘要匹配的 pose ONNX/TensorRT engine 与 OSNet ONNX，再准备完整身份服务：

```powershell
# 在 visionflow-inference 根目录；路径替换为自己有权使用的制品
python gpu_demo/prepare_identity.py --source-repository C:/models/authorized/model_repository --osnet-onnx C:/models/authorized/osnet/model.onnx
# 后续可加 --offline 复用已准备的最终模型目录

docker build -f gpu_demo/Dockerfile -t visionflow-gpu-demo:24.12 .
$env:TRITON_GPU_IMAGE = 'visionflow-gpu-demo:24.12'
docker compose -f gpu_demo/compose.identity.yaml up -d
```

准备器拒绝未知摘要的制品。TensorRT engine 必须匹配目标 GPU 和运行库，不能用改名替代构建与验证。该示例不包含完整模型二进制，也不保证干净机器只执行 Compose 就能完成模型准备；制品要求和许可快照以推理仓说明为准。独立项目名为 `vf-open-gpu-identity`，HTTP/gRPC/metrics 仅绑定本机18010/18011/18012，不使用业务服务。

模型许可分别记录：YOLO为AGPL-3.0，YuNet为MIT，SFace为Apache-2.0，所用OSNet权重为MIT。训练数据、CUDA/TensorRT、容器和二进制依赖仍适用自身条款；不将整个GPU二进制栈描述为全部开源。本次实际验证使用已安装的兼容镜像，镜像ID见运行清单；干净环境基础镜像构建需要另行验证。

## 2. 使用或重新获取输入视频

仓库保留最终输入 [identity-source-clip.mp4](assets/identity-source-clip.mp4) 与自包含 [来源清单](assets/identity-source-manifest.json)。无需重新下载即可跑演示。需要再生素材时：

```powershell
python -B demo/gpu/assets/prepare_identity_source.py --font C:/Windows/Fonts/NotoSansSC-VF.ttf
```

在 Linux 使用已安装的开源 Noto Sans CJK 字体路径，例如 `--font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`。首次再生需要网络与curl；脚本自动调用 `prepare_source.py --output-dir ...` 取得A，并取得B。源视频byte-range、单人片段和编码日志均保存在 `assets/.cache/`，不发布。最终只写入A/B/A视频与来源清单；人物实名只用于来源署名。

## 3. 实际运行身份视频

在 VisionFlow 根目录，使用已配置依赖的环境：

```powershell
python -B demo/gpu/run.py --identity --body-model osnet_gpu --server-url http://127.0.0.1:18010 --input demo/gpu/assets/identity-source-clip.mp4 --output demo/gpu/identity-run-output
```

也可使用与独立推理服务共享网络的Linux客户端容器：

```powershell
$env:VF_GPU_CLIENT_IMAGE = 'visionflow-gpu-demo:24.12'
docker run --rm --gpus all --network container:vf-open-gpu-identity-triton-gpu-identity-1 --entrypoint python3 -v "${PWD}:/workspace" -w /workspace $env:VF_GPU_CLIENT_IMAGE -B demo/gpu/run.py --identity --body-model osnet_gpu --server-url http://127.0.0.1:8000 --input demo/gpu/assets/identity-source-clip.mp4 --output demo/gpu/identity-run-output
```

源视频30fps，默认stride2，对每个输出帧实际推理，生成15fps、15秒的225帧视频；`--stride 1`可处理全部源帧。播放fps不是模型吞吐。图库在有界、有序提交阶段更新；人脸确认的观察才可更新身份原型，无脸时人形相似只产生候选。

原生模型配置要求GPU实例并保存调用计数；对齐、解码、后处理、图库决策及编码仍有CPU步骤。ONNX Runtime模型不宣称已经禁用所有算子级CPU回退，具体执行范围见推理仓说明与运行证据。

## 4. 固化最终视频与证据

推理完成后，在含Docker CLI、OpenCV及imageio-ffmpeg的本机环境执行：

```powershell
python demo/gpu/finalize_video.py --output demo/gpu/identity-run-output --container vf-open-gpu-identity-triton-gpu-identity-1 --source-manifest demo/gpu/assets/identity-source-manifest.json --model-manifest ../visionflow-inference/gpu_demo/identity-model-manifest.json
```

这一步将实际视频转为浏览器兼容的H.264/yuv420p，完整解码确认帧数，从实际服务容器查询GPU及镜像ID，绑定素材、模型和代码摘要。远程服务不能用客户端本机GPU信息冒充服务端证据。

最终示例保留：

- `identity-run-output/annotated.mp4`：原画面、实际身份叠加与本次登记图库。
- `events.jsonl`：每个输出帧的人形、人脸、向量、相似度与确认状态。
- `manifest.json` / `model-manifest.json`：运行、模型、设备和摘要证据。
- `preview-000.png` / `preview-112.png` / `preview-224.png`：登记A、确认B、复核A的真实视频帧。

重新运行会替换该输出目录的结果；需要保留旧证据时先备份。更新输入视频或其来源清单后，应重新运行并固化对应证据，不将新素材与旧运行清单混用。

## 测试与适用范围

```powershell
python -B -m unittest discover -s tests -p 'test_gpu*.py' -v
```

测试覆盖协议、GPU配置检查、有界并发、有序写出、输入保护和失败清理；真实GPU执行与示例结果另由运行清单证明。人脸使用SFace 128维、五点112×112对齐；人形使用OSNet-AIN 512维、OpenCV双线性128×256 RGB与ImageNet标准化，分别在独立空间归一化比较，不混入旧向量库。

本例是匿名内存图库与公开视频回放，不提供实名核验、跨进程持久化或Kafka/S3/Milvus全系统联调证明。A/B/A是显式人工剪辑，素材无遮挡，不据此宣称自然跨镜头追踪、遮挡恢复或大规模识别准确率。
