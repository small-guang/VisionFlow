# 最终 A / B / A 身份演示素材

[index.html](index.html) 展示 [实际身份视频](identity-run-output/annotated.mp4)，其输入是 [identity-source-clip.mp4](assets/identity-source-clip.mp4)。输入来自两个NASA官方真人访谈，连续动态画面按A5秒 → B5秒 → A5秒剪辑，854×480、30fps、450帧、15秒、无音轨。

输入每段常驻 Source A/B 与人工剪辑标识，没有预置检测框、匿名身份或相似度。切点是人工编辑，不能当作自然跨镜头跟踪；真实人物姓名只用于来源署名，系统使用Demo-P001/P002匿名标签。右侧人脸与人形图库在实际推理时登记，未使用预置手选图库。

## 官方来源与剪辑

- Source A：NASA / Johnson Space Center，Marcos Berrios西语访谈，发布于2021-12-06。[官方记录](https://images.nasa.gov/details/jsc2021m000284_NASA_Astronaut_Candidate_Marcos_Berrios%20-%20Spanish%20Language)。最终第一段使用原访谈20–25秒，第三段使用25–30秒。
- Source B：NASA / Johnson Space Center，Christina Birch访谈，同日发布。[官方记录](https://images.nasa.gov/details/jsc2021m000272_NASA_Astronaut_Candidate_Christina_Birch)。最终第二段使用原访谈30–35秒。

两段从官方720P medium MP4选取，缩至854×480并移除全部音轨，避开原片标题卡。NASA API未给出个人摄影/编辑姓名，所查metadata未注明独立第三方版权或音乐来源。精确原URL、API地址、byte-range缓存摘要、单人片段摘要、编辑帧号及最终文件SHA256均嵌入 [自包含来源清单](assets/identity-source-manifest.json)，不依赖另一个来源JSON或手选原帧文件。

## 使用条款与署名

NASA的[媒体使用规则](https://www.nasa.gov/nasa-brand-center/images-and-media/)说明其媒体在美国一般不受版权保护，可用于事实性教育与信息展示。这不是CC授权，也不表示人物肖像、宣传权或NASA标志不受限制。NASA、Marcos Berrios、Christina Birch及其他相关方不为本项目背书；不得把人物或标志用于未经授权的商业代言或宣传。

素材署名：**NASA / Johnson Space Center**。视频保留NASA媒体适用条款，不因本仓库Apache-2.0许可而被重新许可；独立标注的第三方材料仍适用自身权利。

## 再生与保留边界

```powershell
python -B demo/gpu/assets/prepare_identity_source.py --font C:/Windows/Fonts/NotoSansSC-VF.ttf
```

安装方式见 [README](README.md)。Linux可传Noto Sans CJK的TTF/TTC路径。脚本缺少A时自动调用 `prepare_source.py`；A的单人片段、临时来源清单、B的单人片段、原视频HTTP byte-range及编码日志都写入 `assets/.cache/`。这些缓存可删除，再生时从官方URL重新取得；缓存是源文件前缀，不宣称完整原片归档。

发布素材仅需最终A/B/A输入、来源清单与两个再生脚本。登记参考可直接从最终输入第0帧与第150帧读取；另保留的三阶段示例图来自最终实际识别视频。修改来源清单或输入后必须重新生成与之对应的GPU运行证据。

`diagrams/identity-flow.png` / `.svg` 是身份流程说明，不是运行截图。图形使用开源Noto Sans SC（[SIL OFL 1.1](https://github.com/notofonts/noto-cjk/blob/main/Sans/LICENSE)），不在仓库分发字体二进制。实际确认、人体候选与GPU调用分别以 `identity-run-output/events.jsonl` 和运行清单为准。

本例没有遮挡情节；OSNet专用人形特征只提供候选，最终身份由有效人脸确认。不据本素材推断遮挡追踪、自然跨镜头识别或生产准确率。
