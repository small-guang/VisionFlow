# VisionFlow Kafka Topic 说明

VisionFlow 当前订阅 `PanoramicPersonDetSync`、`PtzPersonDetSync`、`PtzFaceDetSync` 三个输入 topic。三类消息都会走同一套关键帧上下文解析和整图自检测流程；设备侧消息中已有的 S/A/B/C 质量等级、检测框、关键点或位置信息不会作为识别依据。VisionFlow 会基于关键帧整图自行跑人脸检测、人型分割检测、质量评估、识别和向量入库。

消息 value 均为 UTF-8 JSON。

下文标题为新部署的默认输出 Topic。已有部署可通过 `KAFKA_*_RECORD_TOPIC` 环境变量继续指定原有名称；切换名称时需同步下游消费者。输出字段不因 Topic 改名而改变。

输出时序：同帧人物并发准备 crop、人脸匹配和 SBS/TBPS 特征，再按原检测顺序提交检测记录与 ReID 分配。检测记录入队发生在该人物适用的特征推理返回后、ReID 查询前；准备失败、0级或编码失败的样本只提交检测记录。

新生 Face ID/ReID 的识别记录分别等待首次人脸/SBS 向量持久化确认；命中已有身份的记录直接入队，包含命中当前进程 pending 身份的情况。因此按序提交不保证新生记录早于后续命中记录到达 Kafka，各 Topic 之间也没有事务性顺序保证。向量元数据只在相应向量写入成功后生成。

## 0. 输入 topic

订阅列表：

- `PanoramicPersonDetSync`
- `PtzPersonDetSync`
- `PtzFaceDetSync`

topic 名称不改变字段解析规则；所有输入都必须满足下述关键帧字段契约。

`keyFrame`、`keyFrameUrl` 是输入 JSON 的兼容字段名，不是订阅 Topic。

VisionFlow 支持从以下字段解析关键帧对象存储地址，按优先级读取：

- `keyframe`
- `keyFrame`
- `keyframeUrl`
- `keyFrameUrl`
- `imageUrl`
- `image_url`

`panoKeyframeUrl` 是可选的第二图片字段，不参与上述主关键帧候选优先级。主关键帧成功处理后，VisionFlow 才会独立解析并处理全景图；字段缺失、为空或规范化后与主关键帧地址相同时跳过。全景图解析、下载或推理异常仅记录日志，不影响已经完成的主关键帧结果。

当前不执行设备 SN 白名单过滤，设备 SN 不限制消息处理。

输入先由 consumer 解析并入本地队列。一个分区批次全部入队后，consumer 就提交该批次最后一个 offset，不等待 worker 完成图片下载、推理或结果写入。JSON/关键帧上下文无效的消息会在入队前跳过；通过预校验的消息不会仅因 SN 在 worker 中被跳过。跳过的无效消息不会产生输出记录，且已提交 offset 后不会依靠 Kafka 自动重放。

推荐字段：

- 设备 SN：类型 `str`，字段名 `sn`；兼容 `devSn`、`deviceSn`
- 事件时间戳：类型 `int`，字段名 `timestamp`，兼容 `time`、`ts`，单位毫秒；缺失时由 VisionFlow 使用当前时间兜底
- 关键帧对象存储地址：类型 `str`，字段名 `keyframe`
- 全景关键帧对象存储地址：类型 `str`，字段名 `panoKeyframeUrl`，可选；地址规则与主关键帧一致
- 关键帧业务 ID：类型 `str`，字段名 `uId`，必须为非空字符串，推荐使用 UUID；VisionFlow 当前不校验 UUID 格式，输出时映射为 `keyframeId`

主关键帧和全景图共享输入 `uId`，因此输出记录的 `keyframeId` 相同；VisionFlow 为两张图分别生成 `pictureVectorId`。全部五类输出固定包含布尔字段 `isPanoKeyframe`：主关键帧为 `false`，`panoKeyframeUrl` 对应的全景图为 `true`。

关键帧地址规则（主图和全景图使用同一规则）：

- `STORAGE_BACKEND=oss` 为默认模式：如果传入 URL，会移除 scheme、host 和 query，并对 path 做 URL 解码，只保留 object path，保持已有 OSS URL 的解析行为。
- `STORAGE_BACKEND=minio` 时，完整 URL 必须是与 `MINIO_ENDPOINT`、`MINIO_BUCKET_NAME` 匹配的路径式地址，例如配置 endpoint 为 `minio-host:9000`、bucket 为 `visionflow-images` 时，`http://minio-host:9000/visionflow-images/sn/picture/frame.jpg` 解析为 `sn/picture/frame.jpg`。不匹配的 endpoint 或 bucket 会拒绝；query 不进入 object key。
- 裸 object key 不移除 bucket 同名首段；例如 `visionflow-images/picture/frame.jpg` 仍按该完整 key 读取。只有匹配的 MinIO 完整 URL 会移除路径中的 bucket 段。
- 如果值已是 `.jpg` 或 `.jpeg` 结尾，按完整 object name 使用。
- 如果值包含 `/` 但没有图片后缀，补成 `{object_name}.jpg`。
- 如果值不含 `/`、没有图片后缀且存在设备 SN，补成 `{sn}/picture/{keyframe}.jpg`。
- 如果值不含 `/`、没有图片后缀且不存在设备 SN，补成 `{keyframe}.jpg`。
- 消息没有显式 SN 时，VisionFlow 会尝试从最终 object name 的第一段推断设备 SN；短名称无法推断时 SN 为空。

原图读取和人脸、人物裁图写入使用同一个配置后端。以下输出中的 `keyframeUrl`、`faceUrl`、`crop_url` 及 Milvus 图片元数据保存 object key，不额外拼接 bucket 或存储主机，也不生成签名 URL；字段名和 object key 命名规则不因切换后端而改变。上游上传和下游读取服务须同步使用相同存储配置；切换需重启 VisionFlow，不迁移历史图片，不自动回退或双写。配置方式见 [配置说明中的图片对象存储](configuration.md#图片对象存储)。

示例：

```json
{
  "sn": "demo-camera-001",
  "timestamp": 1710000000000,
  "uId": "11111111-1111-4111-8111-111111111111",
  "keyframe": "demo-camera-001/picture/1782802987309.jpg",
  "panoKeyframeUrl": "demo-camera-001/picture/1782802987450.jpg"
}
```

## 朝向字段契约

人脸检测、人脸识别、人型检测和人型识别记录都固定包含字符串字段 `faceOrientation`，公共取值仅为 `front`、`side`、`no_face`、`unknown`。

- `front`：已获得有效三姿态角，且 `abs(yaw) < 20°`。
- `side`：已获得有效三姿态角，且 `abs(yaw) >= 20°`；包含纯侧脸和恰好 `20°` 的边界。
- `no_face`：仅用于人型记录；人物裁剪、实例 mask 与人脸检测响应有效后，检测列表为空，或至少有一个有效 bbox 但所有候选未通过 mask/Pose 归属门控（包括可用 Pose 头部锚点与候选全不匹配）。
- `unknown`：检测响应非 list、非空列表没有任何有效 bbox，以及裁剪、mask、关键点或姿态计算等不可判定异常。

朝向分类只由 `yaw` 决定，`pitch/roll` 不作为 Kafka 字段。独立人脸正常流程不使用 `no_face`；人型 0 级在 crop 和 mask 有效时可仅分析朝向，不做人脸识别。归属门槛和坐标规则见[人脸人型详细规则](人脸人型详细规则.md)。

## 1. 人脸检测记录：`visionflow-face-detection-record`

检测框和关键点由 VisionFlow 模型生成，坐标对应关键帧整图。检测记录包含质量分和等级，质量失败也会尽量输出检测记录。质量分级及前置判断统一见[人脸质量规则](人脸人型详细规则.md#一人脸质量与入库)。

字段：

- 关键帧业务 ID：类型 `str`，字段名 `keyframeId`；值来自输入 `uId`
- 整图向量主键：类型 `int`，字段名 `pictureVectorId`；VisionFlow 内部生成，并作为 `alogPictureClip` 主键
- 人脸朝向：类型 `str`，字段名 `faceOrientation`；仅允许 `front`、`side`、`no_face`、`unknown`，独立人脸记录正常流程不使用 `no_face`
- 全景关键帧来源标志：类型 `bool`，字段名 `isPanoKeyframe`；主关键帧为 `false`，`panoKeyframeUrl` 全景图为 `true`，字段固定存在
- 人脸裁剪图 object key：类型 `str`，字段名 `crop_url`；上传失败时为空
- 设备 SN：类型 `str`，字段名 `devSn`
- 事件时间戳：类型 `int`，字段名 `timestamp`，单位毫秒
- 记录创建时间：类型 `int`，字段名 `createdAt`，单位毫秒，由 VisionFlow 生成
- 检测时间：类型 `int`，字段名 `detectTime`，单位毫秒
- 人脸关键点：类型 `dict`，字段名 `faceKeypoint`，格式 `{"points": [[x, y], ...]}`
- 人脸区域：类型 `dict`，字段名 `faceRect`，格式 `{"x": int, "y": int, "w": int, "h": int}`
- 人脸检测置信度：类型 `float`，字段名 `confidence`；直接使用人脸检测模型原始置信度，通常范围为 `0-1`
- 人脸质量分：类型 `float`，字段名 `faceQualityScore`；几何/前置判断失败或质量模型失败时为 `0`
- 人脸质量等级：类型 `int`，字段名 `faceQualityLevel`；`0` 表示丢弃，`1` 表示仅用于识别，`2` 表示用于建库
- 检测状态：类型 `int`，字段名 `status`；`1` 表示可用于识别或建库，`0` 表示失败或丢弃
- AI 检测记录 ID：类型 `str`，字段名 `aiId`
- 说明：类型 `str`，字段名 `info`

示例：

```json
{
	"keyframeId": "11111111-1111-4111-8111-111111111111",
	"pictureVectorId": 462513696774553600,
	"faceOrientation": "side",
	"isPanoKeyframe": false,
	"crop_url": "demo-camera-001/face_crop/0fde162e-7815-4e63-b71b-d0889d8ad763.jpg",
	"devSn": "demo-camera-001",
	"timestamp": 1782802738389,
	"createdAt": 1782803062579,
	"detectTime": 1782803061772,
	"faceKeypoint": {
		"points": [
			[
				130.91412353515625,
				415.3212585449219
			],
			[
				144.4461669921875,
				416.80859375
			],
			[
				132.33729553222656,
				427.600830078125
			],
			[
				132.40286254882812,
				437.47613525390625
			],
			[
				141.88204956054688,
				437.9897155761719
			]
		]
	},
	"faceRect": {
		"x": 118,
		"y": 395,
		"w": 44,
		"h": 58
	},
	"confidence": 0.7838658690452576,
	"faceQualityScore": 1.0,
	"faceQualityLevel": 0,
	"status": 0,
	"aiId": "4df1f02c-f395-4994-b15b-2600a927cabf",
	"info": "人脸质量分 1.00，质量等级 0，已丢弃"
}
```

## 2. 人脸识别记录：`visionflow-face-recognition-record`

独立人脸通过质量、crop 上传和特征校验后进入身份查询，候选来自 Milvus `alogFace` 与当前进程 pending 样本。`score/vectorHitFaceId` 对应最高单条相似度及其身份；灰区证据不足时 `aiFaceId="0"/faceStatus=0`，保留候选分数和 ID 供诊断。

灰区双样本确认可以输出已有 ID，但不提交当前特征。新生记录等待首次向量持久化确认；具体门槛和入库条件见[身份匹配与入库](人脸人型详细规则.md#三身份匹配与入库)。

检索异常时输出 `aiFaceId="0"`、`faceStatus=0`、`score=0.0`、`vectorHitFaceId=""`，在 `info` 中说明失败，不创建或入库；已生成的检测记录保留。

字段：

- 设备 SN：类型 `str`，字段名 `devSn`
- 关键帧业务 ID：类型 `str`，字段名 `keyframeId`；值来自输入 `uId`
- 整图向量主键：类型 `int`，字段名 `pictureVectorId`；VisionFlow 内部生成，并作为 `alogPictureClip` 主键
- 人脸朝向：类型 `str`，字段名 `faceOrientation`；仅允许 `front`、`side`、`no_face`、`unknown`，独立人脸记录正常流程不使用 `no_face`
- 全景关键帧来源标志：类型 `bool`，字段名 `isPanoKeyframe`；主关键帧为 `false`，`panoKeyframeUrl` 全景图为 `true`，字段固定存在
- 人脸裁剪图 object key：类型 `str`，字段名 `faceUrl`
- 事件时间戳：类型 `int`，字段名 `timestamp`，单位毫秒
- 记录创建时间：类型 `int`，字段名 `createdAt`，单位毫秒，由 VisionFlow 生成
- 识别到的人脸 ID：类型 `str`，字段名 `aiFaceId`
- 相似度分数：类型 `float`，字段名 `score`
- 向量库中最高相似度命中的 face_id：类型 `str`，字段名 `vectorHitFaceId`；无命中时为空字符串
- AI 检测记录 ID：类型 `str`，字段名 `aiId`；对应人脸检测记录中的 `aiId`
- 人脸状态：类型 `int`，字段名 `faceStatus`；`0` 表示未分配 face_id，`1` 表示新生 face_id，`2` 表示命中已有 face_id
- 人脸质量分：类型 `float`，字段名 `faceQualityScore`
- 人脸质量等级：类型 `int`，字段名 `faceQualityLevel`；`1` 表示仅识别，`2` 表示建库
- 说明：类型 `str`，字段名 `info`

示例：

```json
{
	"devSn": "demo-camera-001",
	"keyframeId": "11111111-1111-4111-8111-111111111111",
	"pictureVectorId": 462513739325767680,
	"faceOrientation": "front",
	"isPanoKeyframe": false,
	"faceUrl": "demo-camera-001/face_crop/0fde162e-7815-4e63-b71b-d0889d8ad763.jpg",
	"timestamp": 1782802746345,
	"createdAt": 1782803104758,
	"aiFaceId": "86a0d2f3-7be6-4fd7-a619-625391213ca3",
	"score": 0.7989693284034729,
	"vectorHitFaceId": "86a0d2f3-7be6-4fd7-a619-625391213ca3",
	"aiId": "0a56b493-f006-44af-b6d8-1290290339c1",
	"faceStatus": 2,
	"faceQualityScore": 4.0,
	"faceQualityLevel": 1,
	"info": "质量等级 1，命中已有 face_id，最高相似度 0.7990"
}
```



## 3. 人型检测记录：`visionflow-person-detection-record`

人型检测记录描述原图中的人物框、COCO-17 关键点、质量结果和最终 crop 地址。`personOrientation` 表示身体朝向，与 `faceOrientation` 独立；两类人型记录复用同一次身体朝向判定。

crop 允许可变宽高比，整图框和关键点保留原坐标。质量分级、转正、mask/Pose 归属及身体朝向计算见[人型质量规则](人脸人型详细规则.md#二人型质量与入库)。

字段：

- 关键帧业务 ID：类型 `str`，字段名 `keyframeId`；值来自输入 `uId`
- 整图向量主键：类型 `int`，字段名 `pictureVectorId`；VisionFlow 内部生成，并作为 `alogPictureClip` 主键
- 人脸朝向：类型 `str`，字段名 `faceOrientation`；取值和异常含义见[朝向字段契约](#朝向字段契约)
- 人型躯干朝向：类型 `str`，字段名 `personOrientation`；每条人型检测记录固定输出，仅允许 `front`、`side`、`back`、`unknown`。描述躯干/身体方向，独立于 `faceOrientation`（身体 `front`、脸部 `side` 可以同时出现）；由 COCO-17 Pose 计算，分类失败时为 `unknown`
- 全景关键帧来源标志：类型 `bool`，字段名 `isPanoKeyframe`；主关键帧为 `false`，`panoKeyframeUrl` 全景图为 `true`，字段固定存在
- 人型裁剪图 object key：类型 `str`，字段名 `crop_url`；上传失败时为空
- 设备 SN：类型 `str`，字段名 `devSn`
- 事件时间戳：类型 `int`，字段名 `timestamp`，单位毫秒
- 记录创建时间：类型 `int`，字段名 `createdAt`，单位毫秒，由 VisionFlow 生成
- 检测时间：类型 `int`，字段名 `detectTime`，单位毫秒
- 人型区域：类型 `dict`，字段名 `peopleRect`，格式 `{"x": int, "y": int, "w": int, "h": int}`
- 人型关键点：类型 `dict`，字段名 `personKeypoint`，格式 `{"format":"coco17","points":[[x,y,score],...]}`；坐标为整图坐标，无有效姿态时 `points=[]`
- COCO 17 点顺序：鼻、左眼、右眼、左耳、右耳、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左踝、右踝
- 检测置信度：类型 `float`，字段名 `confidence`；直接使用分割模型原始置信度，通常范围为 `0-1`
- mask 像素数量：类型 `int`，字段名 `maskPixelCount`
- 人型质量分：类型 `float`，字段名 `personQualityScore`
- 人型质量等级：类型 `int`，字段名 `personQualityLevel`；`0` 表示丢弃，`1` 表示仅用于 ReID 识别，`2` 表示用于建库
- 人型质量动作：类型 `str`，字段名 `personQualityAction`；取值 `discard`、`identify_only`、`build_gallery`
- mask 质量分：类型 `float`，字段名 `maskScore`
- crop 质量分：类型 `float`，字段名 `cropScore`
- 姿态结构分：类型 `float`，字段名 `poseScore`，范围 `0-10`
- 可见关键点数：类型 `int`，字段名 `visibleKeypointCount`；可见要求点置信度 `>=0.5` 且坐标有效并位于原图内
- 躯干可见点数：类型 `int`，字段名 `torsoKeypointCount`；统计左右肩、左右髋四点，范围 `0-4`
- 完整肩髋链数：类型 `int`，字段名 `shoulderHipChainCount`；统计左右两侧 shoulder→hip 完整链，范围 `0-2`
- 上半身 Pose 建库资格：类型 `bool`，字段名 `halfBodyPoseEligible`；仅表示满足可见点 `>=11`、两肩两髋可见和双侧肩髋链完整的 Pose 例外，其他门控仍可能使最终等级低于2级
- 人体完整度：类型 `str`，字段名 `bodyCompleteness`；取值 `full_body`、`three_quarter_upper`、`three_quarter_lower`、`half_body_upper`、`half_body_lower`、`partial`、`unknown`
- 半身标记：类型 `bool`，字段名 `isHalfBody`；`half_body_upper`、`half_body_lower` 时为 `true`。高质量 `half_body_upper` 可以达到2级；具体条件见人型质量规则
- 确认截断标记：类型 `bool`，字段名 `isTruncated`；仅当 bbox、原图 mask 与必要的 Pose 证据共同确认原图人体被截断时为 `true`。靠近原图边或 mask 触达检测 bbox 不会单独设置该字段
- 总留白比例：类型 `float`，字段名 `paddingRatio`；最终 crop 中无法从旋转前 `5%` 自然窗口取得真实像素、由均值色提供的像素占比。它以原图范围内自然窗口建立 source-validity mask，使用与图像相同的仿射变换和最终 crop 坐标（成功转正时包含紧裁剪平移），以最近邻方式投影后逐像素计算 `1 - validPixelCount / finalCropPixelCount`，限制在 `[0,1]`
- 比例画布兼容诊断：类型 `float`，字段名 `letterboxPaddingRatio`；检测记录和识别记录固定输出 `0.0`
- 自然场景窗口越界比例：类型 `float`，字段名 `sourceOverflowRatio`；`(naturalWindowArea - inFrameNaturalWindowArea) / naturalWindowArea`，即旋转前 `5%` 自然窗口落在原图外的面积比例，限制在 `[0,1]`，只作诊断
- 原始检测框内容越界比例：类型 `float`，字段名 `bboxOverflowRatio`；`1 - clippedRawBBoxArea / rawBBoxArea`，限制在 `[0,1]`；原始 `xyxy` 缺失、非法或非有限时安全归零
- 靠近原图边数量：类型 `int`，字段名 `nearImageEdgeCount`；使用 `max(8px, 原图短边×2%)` 诊断带统计，范围 `0-4`，不直接降级
- 确认截断边数量：类型 `int`，字段名 `truncatedEdgeCount`；范围 `0-4`，达到3条时丢弃
- 外来人物总占比：类型 `float`，字段名 `foreignPersonRatio`；有效外来 mask 并集像素 / 目标 mask 像素
- 外来人物核心区占比：类型 `float`，字段名 `foreignPersonCoreRatio`；目标 mask 外接框中央70%宽度、完整目标高度内的有效外来 mask 像素 / 目标 mask 像素
- 外来人物像素数：类型 `int`，字段名 `foreignPersonPixelCount`；核心区对应字段为 `foreignPersonCorePixelCount`
- bbox 回退诊断：字段名 `foreignPersonBBoxFallbackRatio` 和 `foreignPersonBBoxFallbackPixelCount`；其他人物 mask 缺失、异常或为空时统计，不参与 crop 等级门控
- 边缘主导标记：类型 `bool`，字段名 `foreignPersonEdgeDominant`；至少80%的有效外来 mask 位于核心区之外时为 `true`
- 多人证据来源：类型 `str`，字段名 `foreignPersonEvidenceSource`；取值 `none`、`mask`、`bbox_fallback`、`mixed`
- 多人混入风险标记：类型 `bool`，字段名 `hasSecondPerson`；当前人型框与其他人型框重叠，或者 mask 存在多个较大的连通区域时为 `true`。该字段是当前裁剪图可能混入其他人的启发式判断，不表示已准确识别或统计出第二个人
- 质量门控原因：类型 `list[str]`，字段名 `qualityReasons`；记录降级或丢弃的 reason code
- 检测状态：类型 `int`，字段名 `status`；`1` 表示可用于 ReID 识别或建库，`0` 表示无效或质量不达标
- AI 检测记录 ID：类型 `str`，字段名 `aiId`
- 说明：类型 `str`，字段名 `info`；命中质量门控原因时会追加中文 `门控原因：...`；命中图像预检时还会写入实际指标和阈值

内部 crop 几何诊断 `content_xyxy` 和 `padding` 不属于 Kafka 记录 schema：`content_xyxy=[x1,y1,x2,y2)` 是最终 crop 中有效 source 内容的轴对齐包围框；`padding=(left,top,right,bottom)` 是该包围框到最终 crop 四边的连续合成边界宽度。旋转角区等位于包围框内部的非 source 像素不属于四边连续边界，由逐像素的 `paddingRatio` 表达。

质量门控原因对照表：

说明：`qualityReasons` 是诊断字段，可能同时出现多个 reason code。`硬拒绝` 表示最终 `personQualityLevel=0`、`personQualityAction=discard`，不会提取 SBS R50-IBN/CLIP、不查库、不入库；`建库拦截` 表示最高只能到 `1级`、`personQualityAction=identify_only`，只使用 SBS R50-IBN 识别已有 `reid`，不会生成新 `reid` 或写入 `alogSBS`、`alogTBPS`。人型识别记录只会在 `1级/2级` 通过后产生，因此通常只会看到建库拦截或诊断类原因，完整丢弃原因主要看人型检测记录。

| reason code | 类型/影响 | 触发条件 | 说明 |
| --- | --- | --- | --- |
| `mask_area_too_small` | 硬拒绝 | `maskScore < 5`，等价于 `maskPixelCount < 6000` | mask 面积太小，基础质量分不足识别门槛 |
| `mask_area_below_enroll` | 自然 `1级` | `5 <= maskScore < 7`，等价于 `6000 <= maskPixelCount < 8400` | 可用于识别，但基础 mask 分未达到建库门槛 |
| `invalid_confidence` | 硬拒绝 | `confidence` 为空、NaN、无穷大或 `<= 0` | VisionFlow 不额外设置 `>=0.5` 门槛，正常正值继续进入后续质量门控 |
| `invalid_image_shape` | 硬拒绝 | 原图宽高异常，`image_w <= 0` 或 `image_h <= 0` | 无法可靠计算 bbox/crop 比例 |
| `invalid_bbox` | 硬拒绝 | bbox 为空、面积为 `0`，或 `w <= 0`、`h <= 0` | 无法生成有效人型 crop |
| `empty_crop` | 硬拒绝 | 按 bbox 从原图裁剪后得到空图 | bbox 与图像范围不匹配或裁剪失败 |
| `blurry_laplacian` | 硬拒绝或建库拦截 | `224x224` 前景 Laplacian 方差 `<10` 时硬拒绝，`10 <= value < 20` 时建库拦截 | mask 前景会先去除边缘保护带；生产主流程要求有效 mask，无效 mask 会更早以 `pose_not_matched` 丢弃 |
| `underexposed` | 硬拒绝或建库拦截 | `gray<=5` 占比 `>=0.80` 或 `P95<=20` 时硬拒绝；`gray<=20` 占比 `>=0.70` 或 `P90<=40` 时建库拦截 | 人型前景严重或中度欠曝 |
| `overexposed` | 硬拒绝或建库拦截 | `gray>=250` 占比 `>=0.80` 或 `P5>=245` 时硬拒绝；`gray>=235` 占比 `>=0.70` 或 `P10>=220` 时建库拦截 | 人型前景严重或中度过曝 |
| `low_contrast` | 硬拒绝或建库拦截 | 灰度 `P95-P5 < 8` 时硬拒绝，`8 <= P95-P5 < 20` 时建库拦截 | 人型前景动态范围不足 |
| `bbox_too_small` | 硬拒绝 | `w < 24` 或 `h < 48` | bbox 绝对尺寸过小 |
| `bbox_area_too_small` | 硬拒绝 | `bboxArea / imageArea < 0.002` | bbox 占整图比例过小 |
| `bbox_area_too_large` | 建库拦截 | `bboxArea / imageArea > 0.85` | bbox 几乎覆盖整图，检测可靠性下降 |
| `bbox_aspect_ratio_abnormal` | 建库拦截 | `h / w < 0.8` 或 `h / w > 5.0` | bbox 高宽比不像正常站立/行走人体 |
| `bbox_three_image_edges` | 兼容保留 | 服务不产生此值 | 保留枚举；截断由确认的内容截断原因表达 |
| `bbox_near_image_edge` | 诊断 | 边缘 margin 为 `max(8px, min(image_w, image_h) * 0.02)`；靠近任意数量的边均只记录 | 通过 `nearImageEdgeCount` 查看方向数量，不设置 `isTruncated` |
| `bbox_content_overflow_high` | 建库拦截 | 已确认截断且 `bboxOverflowRatio >0.20`、`<=0.50` | 最高 `1级`；恰好 `0.20` 不降级 |
| `bbox_content_overflow_too_high` | 硬拒绝 | 已确认截断且 `bboxOverflowRatio >0.50` | 丢弃；恰好 `0.50` 仍为最高 `1级` |
| `person_truncated` | 诊断 | bbox 与原图 mask 同方向触边；raw bbox 越界直接确认，否则顶部还需缺头部点、底部还需缺完整脚踝链 | 表示已确认原图人体截断，并设置 `isTruncated=true` |
| `person_truncated_severe` | 硬拒绝 | 确认截断方向达到3条及以上 | 与越界面积无关，直接丢弃 |
| `bbox_heavily_overlaps_other_person` | 硬拒绝 | 与其他 person bbox 的 `intersection/currentArea > 0.40` | 多人严重重叠，当前 crop 可能混人 |
| `bbox_overlaps_other_person` | 建库拦截 | 与其他 person bbox 的 `intersection/currentArea > 0.15` | 多人轻中度重叠，最多只用于识别 |
| `mask_missing` | 硬拒绝 | segment 中没有 `mask` | Mask–Pose 绑定不会生成候选，最终以 `pose_not_matched` 丢弃 |
| `mask_shape_invalid` | 硬拒绝 | mask 不是二维数组，或尺寸不等于原图尺寸 | Mask–Pose 绑定不会生成候选，最终以 `pose_not_matched` 丢弃 |
| `mask_bbox_ratio_too_low` | 硬拒绝 | `maskArea / bboxArea < 0.12` | bbox 里有效人体 mask 占比过低 |
| `mask_bbox_ratio_low` | 建库拦截 | `0.12 <= maskArea / bboxArea < 0.25` | mask/bbox 占比较低，最多只用于识别 |
| `mask_fragmented` | 硬拒绝 | 最大连通域面积 / `maskArea < 0.55` | mask 过于破碎，不像单个完整人体 |
| `mask_multiple_components` | 建库拦截 | 存在 2 个及以上大连通域；大连通域阈值为 `max(64, maskArea * 0.12)` | 可能混入多个人或背景块 |
| `mask_touching_bbox_edge_severe` | 诊断 | mask 触达至少 3 条 bbox 边且边缘带占比 `>= 0.40` | 只描述 mask 与检测框关系，不降级、不丢弃、不设置 `isTruncated` |
| `mask_touching_bbox_edge` | 诊断 | mask 触达 bbox 边或边缘带占比达到诊断阈值 | 只描述 mask 与检测框关系，不参与等级门控 |
| `crop_padding_ratio_high` | 兼容保留 | 服务不产生此值 | 保留枚举；padding 比例仅用于诊断 |
| `crop_padding_ratio_too_high` | 兼容保留 | 服务不产生此值 | 保留枚举；padding 比例仅用于诊断 |
| `crop_contains_other_person_mild` | 诊断 | 达到多人诊断线，但总占比 `<0.10` 且核心区占比 `<0.05` | 边缘轻度混入，不单独降级 |
| `crop_other_person_bbox_fallback` | 诊断 | 其他人物 mask 缺失、异常或为空，bbox 回退达到多人诊断线 | 只记录 bbox 推测，不参与 crop 等级门控 |
| `crop_contains_other_person` | 建库拦截 | 有效外来 mask 总占比 `>=0.10` 或核心区占比 `>=0.05` | crop 混入其他人物，最高 `1级` |
| `crop_heavily_contains_other_person` | 硬拒绝 | 有效外来 mask 总占比 `>=0.30` 或核心区占比 `>=0.20` | crop 严重混入其他人物 |
| `pose_inference_failed` | 硬拒绝 | Pose Triton 请求异常 | 保留检测记录，但不执行 TBPS、ReID 或入库 |
| `pose_not_matched` | 硬拒绝 | 没有同时满足 bbox IoU `>=0.3`、mask 一致率 `>=0.60` 的一对一姿态候选，或 mask 无效 | 分割人型缺少可信姿态归属 |
| `pose_keypoints_invalid` | 硬拒绝 | 关键点不是有效的 `[17,3]` 数组 | 无法计算姿态结构分 |
| `pose_visible_keypoints_too_few` | 硬拒绝 | 可见关键点少于 `6` | 人体结构信息不足 |
| `pose_torso_incomplete` | 硬拒绝 | 左右肩、左右髋四点中可见点不超过 `1` | 躯干结构严重缺失 |
| `pose_shoulder_hip_chain_incomplete` | 建库拦截 | 左右两侧均无完整肩髋链路 | 最高只用于识别 |
| `pose_lower_body_incomplete` | 建库拦截或诊断 | 左右两侧均无完整髋膝链路 | 通常最高只用于识别；合格 `half_body_upper` 中只记录诊断 |
| `pose_half_body` | 建库拦截或诊断 | 判断为 `half_body_upper` | 可见点 `>=11`、两肩两髋可见、双侧 shoulder→hip 链完整且无其他降级/拒绝时只诊断并可达到2级 |
| `pose_head_incomplete` | 建库拦截 | 身体链完整但头部点均不可见，判断为 `three_quarter_lower` | 最高只用于识别 |
| `pose_body_type_not_searchable` | 硬拒绝 | 判断为 `half_body_lower` 或 `partial` | 不提取特征、不查库、不入库 |
| `pose_keypoints_outside_bbox` | 建库拦截 | 可见点落入分割框的比例 `<0.70` | Pose 与 segmentation 框不一致 |
| `pose_mask_mismatch` | 建库拦截 | 有效 mask 存在且可见点落入 mask 的比例 `<0.60` | Pose 与前景分割不一致 |
| `pose_geometry_unstable` | 建库拦截 | 可见点纵向跨度 / bbox 高度 `<0.35` | 点位分布塌缩或局部化 |

示例：

```json
{
	"keyframeId": "d586dc1b-40b5-7917-032e-1162209bf20d",
	"pictureVectorId": 466152308904820736,
	"faceOrientation": "no_face",
	"personOrientation": "front",
	"isPanoKeyframe": false,
	"crop_url": "demo-camera-001/person_crop/c06806f8-6ef9-4bfe-90ea-7ef539c1dd3b.jpg",
	"devSn": "demo-camera-001",
	"timestamp": 1783670571808,
	"createdAt": 1783670576302,
	"detectTime": 1783670576233,
	"peopleRect": {
		"x": 386,
		"y": 405,
		"w": 233,
		"h": 261
	},
	"personKeypoint": {
		"format": "coco17",
		"points": [
			[
				519.4463500976562,
				464.45037841796875,
				0.08758574724197388
			],
			[
				512.4719848632812,
				455.7772216796875,
				0.009224116802215576
			],
			[
				518.6763305664062,
				455.69317626953125,
				0.09847527742385864
			],
			[
				461.3120422363281,
				453.7860107421875,
				0.26506954431533813
			],
			[
				509.8325500488281,
				456.9969177246094,
				0.9000096917152405
			],
			[
				415.607666015625,
				503.55133056640625,
				0.9718165993690491
			],
			[
				508.07177734375,
				507.07550048828125,
				0.9956212639808655
			],
			[
				412.8370056152344,
				576.8698120117188,
				0.367176353931427
			],
			[
				544.0066528320312,
				580.8032836914062,
				0.981255829334259
			],
			[
				490.1474914550781,
				570.49169921875,
				0.1505027413368225
			],
			[
				595.4845581054688,
				564.6778564453125,
				0.8844181299209595
			],
			[
				412.7963562011719,
				643.499267578125,
				0.5699616074562073
			],
			[
				478.899169921875,
				649.431884765625,
				0.7225759029388428
			],
			[
				473.1408386230469,
				666.4466552734375,
				0.012876391410827637
			],
			[
				556.763916015625,
				664.1505126953125,
				0.036016106605529785
			],
			[
				475.4205017089844,
				713.4012451171875,
				0.003077685832977295
			],
			[
				547.483154296875,
				707.4820556640625,
				0.005295813083648682
			]
		]
	},
	"confidence": 0.5257782936096191,
	"maskPixelCount": 29384,
	"personQualityScore": 7.83235294117647,
	"personQualityLevel": 1,
	"personQualityAction": "identify_only",
	"maskScore": 10.0,
	"cropScore": 10.0,
	"poseScore": 6.0588235294117645,
	"visibleKeypointCount": 7,
	"bodyCompleteness": "half_body_upper",
	"isHalfBody": true,
	"isTruncated": false,
	"paddingRatio": 0.42,
	"letterboxPaddingRatio": 0.0,
	"sourceOverflowRatio": 0.09,
	"bboxOverflowRatio": 0.0,
	"nearImageEdgeCount": 1,
	"truncatedEdgeCount": 0,
	"hasSecondPerson": false,
	"foreignPersonRatio": 0.0,
	"foreignPersonCoreRatio": 0.0,
	"foreignPersonPixelCount": 0,
	"foreignPersonCorePixelCount": 0,
	"foreignPersonBBoxFallbackRatio": 0.0,
	"foreignPersonBBoxFallbackPixelCount": 0,
	"foreignPersonEdgeDominant": false,
	"foreignPersonEvidenceSource": "none",
	"qualityReasons": [
		"pose_lower_body_incomplete",
		"pose_half_body"
	],
	"status": 1,
	"aiId": "02ac727e-ec66-474b-bba5-4c6f5e2ca114",
	"info": "人型质量分 7.83，质量等级 1，已通过，mask像素数 29384，pose分 6.06，可见关键点 7，完整度 half_body_upper，门控原因：下半身关键点不完整、检测为半身或局部人体"
}
```

## 4. 人型识别记录：`visionflow-person-recognition-record`

最终质量 1/2 级且编码成功的人物进入特征处理和身份提交。SBS 特征无效时输出 `SKIPPED`；有效时根据当前观察人脸选择同脸或匿名查询。只有“2 级 + 当前非零人脸”才允许创建或扩充 SBS，匿名人物只复用已有 ReID。

匿名流程不调用 enrollment，其状态与原因为空；继承的 `canonicalFaceId` 不授权入库。新生 ReID 记录等待至少一个正式 SBS 原型持久化确认，已有身份命中记录直接入队。TBPS 写入独立判断，SBS 失败时不以 CLIP 分配身份。

查询矩阵、支持数、分差和同帧排除规则统一见[人型 ReID 识别完整流程](人型ReID识别完整流程.md)；本节定义输出字段。

字段：

- 设备 SN：类型 `str`，字段名 `devSn`
- 关键帧业务 ID：类型 `str`，字段名 `keyframeId`；值来自输入 `uId`
- 整图向量主键：类型 `int`，字段名 `pictureVectorId`；VisionFlow 内部生成，并作为 `alogPictureClip` 主键
- 人脸朝向：类型 `str`，字段名 `faceOrientation`；仅允许 `front`、`side`、`no_face`、`unknown`；含义与人型检测记录一致
- 人型躯干朝向：类型 `str`，字段名 `personOrientation`；每条人型识别记录固定输出，仅允许 `front`、`side`、`back`、`unknown`；含义与人型检测记录一致，独立于 `faceOrientation`
- 全景关键帧来源标志：类型 `bool`，字段名 `isPanoKeyframe`；主关键帧为 `false`，`panoKeyframeUrl` 全景图为 `true`，字段固定存在
- 关键帧 object key：类型 `str`，字段名 `keyframeUrl`
- 人型裁剪图 object key：类型 `str`，字段名 `crop_url`；上传失败时为空
- 人脸 ID：类型 `str`，字段名 `aiFaceId`；人型裁剪图中通过 mask 与 Pose 归属门控的人脸 ID，否则为 `"0"`
- ReID：类型 `str`，字段名 `reid`；未分配时为 `"0"`
- 过滤掉本帧不可复用 ReID 后的单条最高原始候选 ReID：类型 `str`，字段名 `vectorHitReid`；无原始候选时为空字符串。它是诊断字段，不保证等于最终类级胜出的 `reid`
- ReID 分数：类型 `float`，字段名 `score`；成功分配时为胜出类的 `0.60×Top1 + 0.40×TopK平均分`，未分配时为过滤后的单条最高原始相似度
- 检测时间：类型 `int`，字段名 `detectTime`，单位毫秒
- 事件时间戳：类型 `int`，字段名 `timestamp`，单位毫秒
- 记录创建时间：类型 `int`，字段名 `createdAt`，单位毫秒，由 VisionFlow 生成
- 人型区域：类型 `dict`，字段名 `peopleRect`，格式 `{"x": int, "y": int, "w": int, "h": int}`
- 说明：类型 `str`，字段名 `info`；发生帧内互斥过滤时会追加“同帧互斥已排除 N 个 reid”，完整排除列表只写 Worker 调试日志
- 人型状态：类型 `int`，字段名 `reidStatus`；`0` 表示未分配，`1` 表示新建有脸 ACTIVE reid，`2` 表示命中 ACTIVE reid
- ReID 生命周期状态：类型 `str`，字段名 `reidState`；只允许空字符串（未分配）或 `ACTIVE`（已有正式身份或当前由观察人脸创建的新身份）
- 当前 SBS 扩充样本状态：类型 `str`，字段名 `reidEnrollmentState`；只允许空字符串、`FORMAL_PENDING` 或 `REJECTED`
- 当前 SBS 扩充原因：类型 `str`，字段名 `reidEnrollmentReason`；直接输出本次 `SbsEnrollmentOutcome.reason`，未尝试扩充时为空字符串

以下 ReID 分配诊断字段在每条人型识别记录中始终存在：

- 分配动作：类型 `str`，字段名 `reidDecision`；取值 `MATCH_EXISTING`、`CREATE_NEW`、`UNASSIGNED`、`SKIPPED`
- 决策原因：类型 `str`，字段名 `reidDecisionReason`；使用稳定 reason code，例如 `accepted`、`unique_face_anchor`、`ambiguous_margin`、`no_compatible_class`；SBS 不可用时使用 `sbs_inference_failed`、`sbs_embedding_missing`、`sbs_embedding_invalid` 或 `sbs_embedding_zero`
- 搜索模式：类型 `str`，字段名 `reidSearchMode`；取值 `SAME_FACE`、`ANONYMOUS_GLOBAL`、`NOT_EXECUTED`
- 分配来源：类型 `str`，字段名 `reidSource`；复用已有 ReID 时取 `MILVUS`、`BUFFER` 或 `MILVUS+BUFFER`，新建时取 `NEW`，未分配或跳过时为空字符串
- 是否新建：类型 `bool`，字段名 `isNewReid`
- 规范人脸 ID：类型 `str`，字段名 `canonicalFaceId`；新建已知脸 ReID 时为当前 `aiFaceId`，命中已有 ReID 时继承胜出候选的规范人脸，因此当前 `aiFaceId="0"` 的匿名匹配也可能输出非零值；未分配或跳过时为 `"0"`。该继承值不能替代当前观察人脸触发 SBS enrollment
- 类内最高原型分：类型 `float`，字段名 `reidTop1Score`
- 类内 TopK 平均分：类型 `float`，字段名 `reidMeanScore`
- 类级聚合分：类型 `float`，字段名 `reidClassScore`；计算方式为 `0.60×reidTop1Score + 0.40×reidMeanScore`
- 类级支持数：类型 `int`，字段名 `reidSupportCount`
- 第二兼容类分数：类型 `float`，字段名 `reidRunnerUpScore`
- 实际类间差：类型 `float`，字段名 `reidMargin`
- 要求类间差：类型 `float`，字段名 `reidRequiredMargin`；已知脸查询为 `0.0`，只应用同 faceId 候选的绝对相似度门槛，多个同脸 ReID 取最高 classScore，不因胜出与次优 ReID 近分而拒绝；匿名查询为 `0.08`，未执行查询为 `0`
- 同帧排除数：类型 `int`，字段名 `sameFrameExcludedReidCount`；表示实际传入 Milvus 和 pending 查询的不可复用 ReID 数量

因 `ambiguous_margin` 未分配时，类级诊断字段保留领先类的指标；完全没有兼容类时为零。成功分配时 `score` 为胜出类分数，未分配时为最高原始候选分数；`vectorHitReid` 始终用于最高原始候选诊断。

身份状态和样本状态相互独立：匿名命中 ACTIVE 时为 `2/ACTIVE/空`，匿名未命中时为 `0/空/空`；已知当前人脸首次创建 ReID 时为 `1/ACTIVE/FORMAL_PENDING`。

`ACTIVE` 表示业务身份已经确认，`FORMAL_PENDING` 表示本次样本进入正式写入流程，这两个字段本身仍不是 Milvus 回执。新生 ReID 记录在首次 SBS 写入成功后发布时携带 `ACTIVE/FORMAL_PENDING`；实际向量主键继续以向量元数据 Topic 中对应的 SBS 记录为准。

发布确认和重试在当前进程内协调，不提供跨进程的 Kafka exactly-once 保证；实现边界见[缓冲与输出](代码架构与处理流程.md#84-缓冲与-kafka-输出)。

- 人型质量分：类型 `float`，字段名 `personQualityScore`
- 人型质量等级：类型 `int`，字段名 `personQualityLevel`；`1` 表示仅识别，`2` 表示建库
- 人型质量动作：类型 `str`，字段名 `personQualityAction`；取值 `identify_only` 或 `build_gallery`
- mask 质量分：类型 `float`，字段名 `maskScore`
- crop 质量分：类型 `float`，字段名 `cropScore`
- Pose/几何质量摘要：`poseScore`、`visibleKeypointCount`、`torsoKeypointCount`、`shoulderHipChainCount`、`halfBodyPoseEligible`、`bodyCompleteness`、`isHalfBody`、`isTruncated`、`hasSecondPerson`、全部 `foreignPerson*` 多人污染诊断字段、`paddingRatio`、`letterboxPaddingRatio`、`sourceOverflowRatio`、`bboxOverflowRatio`、`nearImageEdgeCount`、`truncatedEdgeCount`；含义与人型检测记录一致，识别记录不包含原始 `personKeypoint`
- 质量门控原因：类型 `list[str]`，字段名 `qualityReasons`
- AI 检测记录 ID：类型 `str`，字段名 `aiId`

示例：

```json
{
	"devSn": "demo-camera-001",
	"keyframeId": "fe338f21-30a8-81bb-cddc-91ebb759e062",
	"pictureVectorId": 465799814198591488,
	"faceOrientation": "no_face",
	"personOrientation": "back",
	"isPanoKeyframe": false,
	"keyframeUrl": "demo-camera-001/picture/1783586528497.jpg",
	"crop_url": "demo-camera-001/person_crop/c06806f8-6ef9-4bfe-90ea-7ef539c1dd3b.jpg",
	"aiFaceId": "0",
	"reid": "0",
	"vectorHitReid": "",
	"score": 0.0,
	"detectTime": 1783586530135,
	"timestamp": 1783586528728,
	"createdAt": 1783586530364,
	"peopleRect": {
		"x": 869,
		"y": 370,
		"w": 308,
		"h": 315
	},
	"info": "质量等级 2，SBS 未命中人型向量，未生成新 reid，最高相似度 0.0000，无脸人体采用严格类级检索；未找到属于当前人型的人脸；人脸归属失败：0 张有效检测均不满足人物mask中心点和覆盖率>=0.50",
	"reidStatus": 0,
	"reidState": "",
	"reidEnrollmentState": "",
	"reidEnrollmentReason": "",
	"reidDecision": "UNASSIGNED",
	"reidDecisionReason": "no_compatible_class",
	"reidSearchMode": "ANONYMOUS_GLOBAL",
	"reidSource": "",
	"isNewReid": false,
	"canonicalFaceId": "0",
	"reidTop1Score": 0.0,
	"reidMeanScore": 0.0,
	"reidClassScore": 0.0,
	"reidSupportCount": 0,
	"reidRunnerUpScore": 0.0,
	"reidMargin": 0.0,
	"reidRequiredMargin": 0.08,
	"sameFrameExcludedReidCount": 0,
	"personQualityScore": 10.0,
	"personQualityLevel": 2,
	"personQualityAction": "build_gallery",
	"maskScore": 10.0,
	"cropScore": 10.0,
	"poseScore": 9.5,
	"visibleKeypointCount": 17,
	"bodyCompleteness": "full_body",
	"isHalfBody": false,
	"isTruncated": false,
	"paddingRatio": 0.3333333333,
	"letterboxPaddingRatio": 0.0,
	"sourceOverflowRatio": 0.0,
	"bboxOverflowRatio": 0.0,
	"nearImageEdgeCount": 0,
	"truncatedEdgeCount": 0,
	"hasSecondPerson": false,
	"foreignPersonRatio": 0.0,
	"foreignPersonCoreRatio": 0.0,
	"foreignPersonPixelCount": 0,
	"foreignPersonCorePixelCount": 0,
	"foreignPersonBBoxFallbackRatio": 0.0,
	"foreignPersonBBoxFallbackPixelCount": 0,
	"foreignPersonEdgeDominant": false,
	"foreignPersonEvidenceSource": "none",
	"qualityReasons": [],
	"aiId": "4af862ad-f2e3-416e-bc4c-bba539ccdfce"
}
```

## 5. 向量信息记录：`visionflow-vector-record`

VisionFlow 在向量成功写入 Milvus 后，会发送向量元数据记录。该消息不包含原始 embedding。

覆盖目标类型：

- `face`：人脸识别向量
- `person`：人型 CLIP 检索向量
- `person_sbs`：人型 SBS R50-IBN ID 分配向量
- `picture`：关键帧整图向量

字段：

- 向量 ID：类型 `int`，字段名 `vectorId`
- 向量维度：类型 `int`，字段名 `vectorDim`；`person_sbs` 为 `2048`，人型 CLIP 的 `person` 为 `512`
- 设备 SN：类型 `str`，字段名 `devSn`
- 关键帧业务 ID：类型 `str`，字段名 `keyframeId`；值来自输入 `uId`
- 整图向量主键：类型 `int`，字段名 `pictureVectorId`；VisionFlow 内部生成，并作为 `alogPictureClip` 主键；当 `targetType=picture` 时 `vectorId == pictureVectorId`
- 全景关键帧来源标志：类型 `bool`，字段名 `isPanoKeyframe`；主关键帧为 `false`，`panoKeyframeUrl` 全景图为 `true`，字段固定存在
- 事件时间戳：类型 `int`，字段名 `timestamp`，单位毫秒
- 记录创建时间：类型 `int`，字段名 `createdAt`，单位毫秒，由 VisionFlow 生成
- 关键帧 object key：类型 `str`，字段名 `keyframeUrl`
- 向量模型名称：类型 `str`，字段名 `modelName`
- 向量数据库名称：类型 `str`，字段名 `vectorDbName`
- AI 检测记录 ID：类型 `str`，字段名 `aiId`；`picture` 类型为空字符串
- 目标类型：类型 `str`，字段名 `targetType`，取值为 `face`、`person`、`person_sbs`、`picture`

模型名称映射：

- `person`：`Person ReID Vector | OpenAI CLIP ViT-B/16 (ViT-B-16.pt)`
- `person_sbs`：`Person ReID Vector | FastReID SBS R50-IBN (DukeMTMC)`
- `picture`：`Picture Keyframe Vector | OFA-Sys Chinese-CLIP ViT-B/16 + RoBERTa-wwm`
- `face`：`Face Recognition Vector | InsightFace buffalo_l (w600k_r50.onnx)`

向量数据库名称映射：

- `person`：`alogTBPS`
- `person_sbs`：`alogSBS`
- `picture`：`alogPictureClip`
- `face`：`alogFace`

示例：

```json
{
  "vectorId": 123456,
  "vectorDim": 512,
  "devSn": "demo-camera-001",
  "keyframeId": "11111111-1111-4111-8111-111111111111",
  "pictureVectorId": 462513696774553600,
  "isPanoKeyframe": false,
  "timestamp": 1710000000000,
  "createdAt": 1710000000456,
  "keyframeUrl": "demo-camera-001/picture/1780279016175.jpg",
  "modelName": "Person ReID Vector | OpenAI CLIP ViT-B/16 (ViT-B-16.pt)",
  "vectorDbName": "alogTBPS",
  "aiId": "b7e6a5b8-7e4a-4645-9c9c-6ebe547833a4",
  "targetType": "person"
}
```
