# CPU 人脸检测演示

此演示将图片发送到 `visionflow-inference/cpu_demo` 的 HTTP 服务，并输出检测框图片和 JSON。它使用 OpenCV 自带的 Haar 级联分类器，无需 GPU 或另行下载模型；这是一个独立的轻量示例，不使用生产 Worker、Kafka、Milvus，也不实现 Triton 协议。

## 使用 Docker 启动推理服务

分别获取 `VisionFlow` 和 `visionflow-inference` 两个仓库。在推理仓库执行：

```bash
docker compose -f cpu_demo/compose.yaml up --build
```

服务只将端口绑定在本机 `127.0.0.1:8080`。此演示没有认证或 TLS，不要直接暴露到公网。

在 VisionFlow 仓库使用 Python 3.12 或 3.13 创建独立虚拟环境，安装演示依赖，然后运行：

```bash
python -m pip install -r demo/requirements.txt
python demo/client.py demo/assets/astronaut.png --output demo/run-output
```

结果保存为 `demo/run-output/detected.jpg` 和 `demo/run-output/detections.json`。也可以将输入路径换成自己有权使用的照片。检测分数是分类器的原始权重，不是概率。停止服务使用 `docker compose -f cpu_demo/compose.yaml down`。

## 不使用 Docker

在推理仓库的独立虚拟环境中安装 `cpu_demo/requirements.txt`，然后执行 `python cpu_demo/server.py`；再按上面的客户端步骤操作。两端可运行在同一个 Python 环境，也可分别运行。

重新生成 README 效果图，在 VisionFlow 仓库执行：

```bash
cd demo
python create_preview.py
```

预览从 `run-output` 的实际结果生成；更换输入图片时，可用 `--input` 指定同一张图片。演示结果会随图片和 OpenCV 版本变化，不是生产模型的性能指标。

## 图片与模型来源

- `assets/astronaut.png`：NASA 的 Eileen Collins 照片，经 [scikit-image 示例数据](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)分发；该文档标注为公共领域。图片中的 NASA 标识不表示 NASA 对本项目的支持。
- 检测器随 OpenCV Python 包提供；`haarcascade_frontalface_default.xml` 文件内包含其 Intel 开源许可和版权声明。它与项目主流水线使用的模型无关。
