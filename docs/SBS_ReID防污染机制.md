# SBS ReID 防污染机制

SBS 使用 2048 维人物特征分配 ReID。防污染规则约束查询身份、同帧共享和原型更新；完整阈值、状态表及集合结构统一见[人型 ReID 识别完整流程](人型ReID识别完整流程.md)。

## 身份边界

| 情况 | 约束 |
| --- | --- |
| 当前观察到可靠人脸 | 只查询相同规范 Face ID 的类；未命中时不改查匿名类或其他人脸 |
| 当前没有可靠人脸 | 使用严格匿名查询，只复用已有 ACTIVE ReID，不创建或扩充 SBS |
| 从 ReID 继承了 `canonicalFaceId` | 仅描述命中身份，不能替代当前观察人脸授权入库 |
| 同一 ReID 含多个非零 Face ID | 冲突类不能成为最终身份候选 |
| 同帧不同人物 | 已占用 ReID 同时从 Milvus 和 pending 查询排除；仅满足 bbox 与 mask 双重重复检测条件时允许共享 |
| SBS 特征无效 | 跳过 ReID 分配；TBPS/CLIP 不承担身份回退 |

同脸查询和匿名查询的绝对分数、支持数、身份分差规则见[类级解析](人型ReID识别完整流程.md#9-类级解析)。同帧排除在身份锁内、查询前完成，人物按原检测顺序提交结果。锁外只预取候选；提交时重新加锁、复核排除和本进程写入版本，并读取最新 pending，失效则完整重查。强一致查询后跨 Milvus/pending 的同一原型按稳定 UUID 只计一次支持，不能因异步通知尚未完成而把一个原型当作两个。

## 原型写入

SBS enrollment 必须同时满足：人物最终 2 级、当前观察到非零 Face ID、有效 SBS 特征，以及非空且 ACTIVE 的 ReID。匿名正常流程跳过 enrollment；直接向 SBS buffer 提交匿名样本时返回 `REJECTED/face_required`。

原型更新拒绝人脸冲突和离群样本，跳过重复样本，并限制每个 ReID 及“设备 × 身型”域的容量。达到上限后仅替换同域冗余原型，无安全替换项或删除失败时冻结更新。数值门槛统一见[有界原型策略](人型ReID识别完整流程.md#12-有界原型策略)。

人物 crop 上传失败时，`cropUrl` 为空，符合条件的 enrollment 可继续；不以整帧地址代替人物裁剪地址。TBPS 独立写入，条件是最终 2 级且 CLIP 特征有效。

## 持久化与记录发布

新生 ReID 识别记录等待该身份至少一个正式 SBS 原型持久化确认。pending 原型使用稳定 UUID；回执超时后重查同一主键，失败时原型和记录一起重试。

`SbsRecognitionRelease` 协调记录绑定与持久化确认，在当前进程内只放行一次。已有身份命中记录直接入队，可能早于新生记录到达；`ACTIVE/FORMAL_PENDING` 表示身份与样本状态，不是当前原型的写入回执。该机制不提供跨进程、跨 Kafka Topic 的事务保证。

## 排查入口

| 问题 | 入口 |
| --- | --- |
| 人脸归属或身份不可靠 | [人脸质量和匹配规则](人脸人型详细规则.md)；`utils/reid/person_face_recognition.py` |
| 候选混入其他身份或同帧人物 | `utils/reid/person_search.py`、`reid_resolution.py`、`person_identity.py` |
| 样本重复、离群或容量受限 | `utils/reid/prototype_policy.py` |
| 新生记录未发布或重复重试 | `utils/buffers/sbs_feature_buffer.py` |
| 集合结构校验失败 | [SBS Schema](人型ReID识别完整流程.md#13-alogsbs-schema)；`db/zilliz/SBS_feature.py` |

输出诊断字段及 reason code 见 [Kafka Topic 契约](kafka_topic.md)。
