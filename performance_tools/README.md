# Kafka 回放与链路耗时观察

## 压测窗口综合采样

`observe_benchmark.py` 在 `observe_io.py` 的只读资源采样基础上，同时采集 Milvus、etcd 的 Prometheus 指标和已有 ALog 性能日志，不发送压测流量、不导入业务连接配置、不启动或重启服务。可在外部发流前启动采集，覆盖发流及排空过程：

```bash
python performance_tools/observe_benchmark.py \
  --output /tmp/alog-benchmark-20261103-1000 \
  --seconds 300 --interval 5 --resource-interval 1 \
  --root alog=1234 --root milvus=5678
```

把示例 PID 改为对应的**主机 PID**；`--root` 可省略或重复，用于跟踪各根进程及其子进程的 CPU、RSS 和 I/O。默认读取 `logs/ALog.log`、`logs/pipeline_stats.jsonl` 和 `http://127.0.0.1:9091/metrics`，路径和 Milvus 地址可分别通过 `--app-log`、`--pipeline-path`、`--milvus-url` 修改。人脸和 SBS 集合默认为 `alogFace`、`alogSBS`；使用隔离前缀时通过 `--face-collection`、`--sbs-collection` 指定完整集合名。输出目录必须不存在；采集器本身会产生少量磁盘写入。默认采集 `sda` 和 `mmcblk0`；新增磁盘可用重复的 `--disk` 额外指定，例如 `--disk nvme0n1`，请求的设备不存在时会明确记录。

etcd 默认不采集；如需在当前 Docker 部署下采集，可每次通过 Docker 查询当前网络地址，再将 `--etcd-url "http://${TASK_ETCD_IP}:2379/metrics"` 加入命令。容器重建后地址可能变化，不应将某次地址写死；Docker 操作权限不足时需使用有权限的账号。

```bash
TASK_ETCD_IP="$(sudo docker inspect --format '{{(index .NetworkSettings.Networks "camtellsvi-milvus").IPAddress}}' milvus-etcd)"
```

结果包括 `benchmark_summary.json`（完整性能汇总桶、ALog 完成速率、队列快照、资源及各指标有效性）、`prometheus.jsonl`（白名单原始指标样本）以及 `resources/` 下的原始资源、pipeline 和采集摘要。`--root` 未提供时不会自动推断各服务进程。采样端点不可达、计数器重置、指标缺失或样本不足不等于零负载，应检查汇总中的原因、实际窗口及原始样本。GPU load 是逐路径 sysfs 原值；Jetson 常见千分比，但无法读取时不推测 GPU 利用率。

ALog 性能日志只汇总完整落入采集窗口的周期，按 PID 展示；pipeline 队列按日志来源、session 和 PID 分组。输入、业务图片完成和异步输出持久化是不同口径；短时快照峰值可能低于真实峰值。Milvus 的集合 Search 耗时、跨集合 Proxy/QueryNode 排队和 etcd fsync 分别使用各自指标时间窗口：Prometheus 直方图分位数为桶上界，不能与 SDK 或其他集合的样本直接逐请求相减。日志轮转、进程重启及其他并发业务也可能影响覆盖或归因；严格比较时应另外记录发流起止、输入组成和排空情况。

验证采集工具本身：

```bash
python -m unittest performance_tools.test_observe_benchmark
python -m unittest performance_tools.test_benchmark_disks
```

## 只读实时 I/O 观测

`observe_io.py` 使用 Python 标准库读取 Linux `/proc`、`/sys` 和已有 pipeline 日志；不导入业务配置、不读取环境变量或密钥、不调用 Docker、不发送 Kafka、不启动服务，也不会向被观察进程发信号。可在用户正在压测时独立运行：

```bash
python performance_tools/observe_io.py \
  --output performance_tools/results/io-observation-20261020-1600 \
  --seconds 180 --interval 1 \
  --pipeline-path logs/pipeline_stats.jsonl \
  --root alog=1234 --root triton=2345 --root minio=3456 \
  --root milvus=4567 --root etcd=5678
```

将示例 PID 换成**主机 PID**，每个 `--root` 名称必须唯一；可重复 `--pipeline-path` 采集其他进程日志。仅在读取其他用户的 `/proc/PID/io` 确实需要时使用具有相应权限的账号运行。权限不足、进程退出或缺少内核指标会记录为降级信息；采集继续进行，不把缺失值当作0。输出目录不能已有本工具同名结果，避免混合不同观察会话；Ctrl+C 只停止采集器。

| 文件 | 内容 |
| --- | --- |
| `metadata.json` | 指定根进程的 PID/start_ticks、采样周期、计数单位和范围 |
| `resources.jsonl` | 每秒整机磁盘/CPU/内存/vmstat/PSI/GPU，指定进程树 `/proc/PID/io`，各容器 cgroup v2 I/O、内存与CPU指标 |
| `pipeline.jsonl` | 已有 pipeline 快照及实际观测时间；首读只保存最新完整快照，随后增量追加 |
| `summary.json` | 生命周期、首末资源/磁盘样本窗口、各pipeline自身窗口、样本数和中断状态 |

进程身份以 PID 与 start_ticks 绑定，避免 PID 复用后把新进程算入原服务；初次及每10秒刷新一次进程树，普通1秒采样不扫描整个 `/proc`。同时采集所有 `/sys/fs/cgroup/system.slice/docker-*.scope`，覆盖未列为根进程的其他容器（例如 Java 服务），以及指定进程所属的 cgroup。`io.stat` 保留各设备 major:minor；父子 cgroup 有累计包含关系，不能直接相加。普通文件中的 cgroup scope/设备编号需要与部署挂载关系另行核对，本工具不调用 Docker 获取服务名。

磁盘默认采集 `sda`/`mmcblk0`，可通过重复 `--disk` 额外指定 `/proc/diskstats` 中的设备名；请求但未发现的设备记录在 `missing_disk_devices`。采集内容包括原始累计计数及相邻样本的读写IOPS、MiB/s、await、flush、队列和util；计数回退、字段结构变化、设备身份缺失/变化或非正窗口不输出有效差值。缺失PID I/O以`io_available=false`标记，cgroup各指标以`availability`标记；对应原始计数字段保持缺失，不能将其补成0。vmstat包含pswpin/out、pgpgin/out、dirty/writeback等；进程RSS单位为页，metadata保留页大小，meminfo单位为KiB。GPU保留sysfs原值及路径，Jetson的load通常为千分比，cur_freq为Hz。日志有界尾读会保留半行、识别轮转/原地重写，并报告超过追赶上限的缺口。

`image_pipeline_completed` 是图像完成计数，可能包含pano；`message_processing.completed_total` 是消息完成计数，两者均不等于异步输出持久化完成。只在相同session和PID、时间有时区且正序的有效窗口内计算速率；任一端缺失所需累计值时窗口invalid，只有显式记录的0才可按0参与计算。

`elapsed_seconds`仅表示采集器生命周期，包含最后一次采集开销，不是吞吐分母。磁盘差值使用实际读取点的`disk_monotonic`；CPU、内存/GPU、PID与cgroup也保留读取点供对齐。`resource_window_seconds`与`disk_window_seconds`是首末采样点间隔，pipeline窗口使用其自身快照时间，起止可能不同；不能把这些未对齐窗口当作严格同窗结果。共享盘的磁盘await是整机观测；容器/PID写字节不能直接解释为该服务的fsync耗时，也不能仅凭util判断瓶颈。应在同一窗口结合输入积压、上传/查询排队、设备延迟及服务指标判断；要证明存储因果收益，还需要固定输入的受控对照。采集器自身也有少量追加写入，每秒刷新Python缓冲但不执行fsync。

该工具的纯单测不连接服务：

```bash
python -m unittest discover -s tests -p "test_io_observer.py"
```

`kafka_replay.py` 用当前 Kafka 的真实关键帧消息重走 ALog 的“下载 → 整图向量 → 人脸 → 人型 → 异步向量写库与 Kafka 输出”链路。默认输入 Topic 为 `PtzPersonDetSync`。脚本不导入 `main.py`、不启动第二个 Worker、不创建 Milvus 客户端，运行中的 ALog 负责处理回放消息。

连接配置统一使用 `core.config.settings`；需要项目现有依赖及能访问 Kafka 的运行环境。推荐在当前 ALog 容器内执行。以下命令在项目根目录执行；容器里项目目录通常是 `/app`，可使用 `docker exec -w /app <alog-container> python ...`。不需要 root 权限的 Python 逻辑，也不需要在命令行填任何连接密码。

## 隔离回放与比较条件

历史快照可以重复使用，但每轮执行都会产生新的业务数据。为测试实例设置独立输入、消费组、五类输出、身份集合和裁图目录，例如在它的环境配置中加入：

```dotenv
KAFKA_INPUT_TOPICS=alog-test-input
KAFKA_GROUP_ID=alog-test-worker
KAFKA_FACE_DETECTION_RECORD_TOPIC=alog-test-face-detection
KAFKA_FACE_RECOGNITION_RECORD_TOPIC=alog-test-face-recognition
KAFKA_PERSON_DETECTION_RECORD_TOPIC=alog-test-person-detection
KAFKA_PERSON_RECOGNITION_RECORD_TOPIC=alog-test-person-recognition
KAFKA_VECTOR_RECORD_TOPIC=alog-test-vector
MILVUS_COLLECTION_PREFIX=alog_test_
STORAGE_OUTPUT_PREFIX=alog-test/crops
```

测试 Worker 和回放工具使用相同的输出 Topic 配置。`replay` 的 `--topic` 默认仍是 `PtzPersonDetSync`，**不会自动跟随 `KAFKA_INPUT_TOPICS`**，隔离运行务必显式添加 `--topic alog-test-input`。先确认测试 Topic 已创建、Worker 已订阅，并确保该实例未订阅生产输入。后文原有命令展示默认 Topic 用法，测试命名空间运行时均应加上该参数。

`MILVUS_COLLECTION_PREFIX` 直接加到四个集合名之前，如 `alog_test_alogFace`，不自动迁移、删除或恢复任何集合。`STORAGE_OUTPUT_PREFIX` 只为新裁图加前缀；原图仍从所选存储后端按历史 object key 读取，Kafka/Milvus 元数据保留实际写入的带前缀 key。换后端不回退历史后端，换输入 Topic 也不自动隔离图库。共用 Triton、Milvus 或存储仍会竞争其容量，不能把混合流量的硬件利用率全归因于本轮。

A/B 两轮固定消息快照及 SHA256、发送顺序/节奏、pano 开关、事件时间策略、算法 TOML、模型版本和有效配置。每轮从同一测试图库初始状态开始，分别观察冷库建档和已有身份命中；不能拿第一轮建库和第二轮复用身份直接比较。保留完整图片完成量、异步输出 ACK/排空时间、各资源排队、RSS 与 CPU/GPU 同窗口采样。只有完整吞吐提高且积压、时延与身份规则仍满足要求，才能确认收益。

本地既有历史快照 `performance_tools/results/qps6-snapshot/snapshot.jsonl` 含 150 条消息，SHA256 为 `7c700240b08c0d8057d8b6de874669c3e54d10a12822d39426057deac37d8b23`（2026-10-19 核对）。它是被 Git 忽略的本地产物，不随仓库分发；使用前重新核验。600 条回放对应循环四遍该快照，并不是 600 张独立图片。历史消息提供字段、SN、事件时间与 pano 语义，但图片可读性、样本身份真值仍需另行验证。

### 当前资源档位与对照记录

代码与两份环境模板统一采用本轮回放后保留的 `baseline` 档位。扩大并发与查询批大小尚未带来稳定的完整吞吐收益，因此保留较小的资源预算；该选择不代表其他机器上的最优配置。

| 资源 | 当前代码与模板默认值 |
| --- | --- |
| 领取 worker / 视觉进程 | `ALOG_WORKER_COUNT=32`、`VISUAL_WORKER_PROCESSES=4` |
| 已接纳消息 / 图片 | `ALOG_MAX_INFLIGHT_MESSAGES=100`、`ALOG_MAX_INFLIGHT_FRAMES=100` |
| 图片 / 存储 / Milvus 查询 / 写入线程 | `IMAGE_CPU_WORKERS=4`、`STORAGE_IO_WORKERS=64`、`MILVUS_SEARCH_WORKERS=16`、`MILVUS_WRITE_WORKERS=4` |
| 存储下载保底 | `STORAGE_DOWNLOAD_RESERVED_WORKERS=4`，包含在存储总额内 |
| Face 查询调度 | `MILVUS_QUERY_BATCH_SIZE=4`、`MILVUS_QUERY_WAIT_MS=1`、`MILVUS_QUERY_CAPACITY=128` |
| Face / Person 准备 | 每帧均为 30，全局均为 100，额外前瞻均为 2 |
| Triton 客户端组批 / 远程 RPC | 目标 B4、等待 2 ms；RPC 容量 80、client 容量 100，已形成 batch 等待区 256 |
| 系统共享内存 | 模板 `TRITON_SYSTEM_SHM_SLOTS=0`、`ALOG_IPC_MODE=private`；本机同机配置为 16 槽 × 32 MiB、两端 `host` IPC |

`VISUAL_WORKER_PROCESSES=0` 仍是有效的单进程模式，可用于受限环境或固定条件下的对照；不是当前默认。多进程的 CPU、存储、Triton 与准备容量均为全局预算，比较时应同时记录每个视觉进程实际分到的份额和 owner 队列。Triton 远程模型与 engine 必须实际支持配置的 B4；Jetson 使用 `orin_gpu` 及新 GPU 合批模块。ALog 的本地 batch、Triton 收到的 request 数和 TensorRT 实际 enqueue 数是不同计数，不能相互代替。

异步 Face 查询始终走同一个 `VectorQueryCoordinator`；将批大小设为 1 只限制每次 RPC 的向量数，不关闭调度器。对照时固定 Strong 一致性、过滤、召回额度、算法阈值和图库初始状态，比较完整图片完成与输出排空，同时核对身份零/非零变化、split/merge、质量字段及错误数。调度时序变化不能只靠随机 ID 重映射判定为等价。

### 本地图片集的补充用途

2026-10-19 对 `datasets/test` 的只读检查得到 87 张 JPEG：71 张为 2560×1440，16 张为 2560×2560，总压缩大小 27,139,363 字节（约 25.88 MiB）；全部解码为三通道 BGR 时约占 1,099,776,000 字节（约 1.024 GiB），不宜一次全部载入。目录没有框、身份、质量或朝向标注。

其中 `0eeadeaaae57d5b3115de899a28c0d3ad85c41c78413f9d57dedb2623fd74c03.jpg` 缺少 JPEG 结束标记：OpenCV 发出截断警告后仍返回图像，Pillow 完整加载报错。主基准应在样本 manifest 中明确排除此图，并单独作为坏图/容错案例；不要静默修补或将容错解码视为完整正常样本。

这组图片适合解码/裁图 CPU、内存调度、有限视觉负载及模型输出结构回归，不能直接替代历史 Kafka 消息，也不能计算身份准确率。后续可使用专用只读本地 loader，或将图片上传测试前缀并构造合约消息；当前回放脚本不会自动把本地目录转换为 Kafka 输入。涉及身份规则的验收仍需有可信标注的多身份序列。

## 1. 只读采样

```bash
python performance_tools/kafka_replay.py snapshot \
  --topic PtzPersonDetSync --count 100 \
  --output performance_tools/results/person_snapshot.jsonl
```

`snapshot` 只读历史，不写 Kafka。它使用 `group_id=None`、关闭自动提交、手动分配分区，不加入生产消费者组、不提交 offset。读取开始时冻结每个分区的 end offset，仅在其前扫描，每分区默认最多回看 1,000 个 offset，总扫描上限 10,000 条，最长 30 秒，不会追读新写入消息。可使用 `--lookback`、`--max-scan`、`--timeout` 调整边界。

仅保存满足项目关键帧字段契约的 JSON object，原始字段和来源 Topic/partition/offset 保留。输出旁的 `.metadata.json` 包含扫描边界、跳过条数和采样量。采样不会下载图片，因此“字段有效”不保证历史图片仍存在；回放后需要检查下载错误与图片完成量。快照文件存在时会拒绝覆盖，重复采样请更换文件名。

## 2. 小规模真实回放

```bash
python performance_tools/kafka_replay.py replay \
  --input performance_tools/results/person_snapshot.jsonl \
  --count 10 --qps 1 --wait-seconds 60 \
  --log-file logs/ALog.log
```

**`replay` 会真实写 Kafka，ALog 随后会执行真实推理、上传裁图、写 Milvus 和输出记录。** 省略限量参数时默认发送 10 条、每秒 1 条；不需要再次交互确认。脚本生成本轮独立 `uId`，保留原 SN、主图片字段和事件时间。相同图片可以多次回放，会产生新的图片向量和业务输出，已有身份仍按现行业务规则处理。

默认去掉 `panoKeyframeUrl`，便于一条输入对应一张主图；`--keep-pano` 保留它，此时每条消息最多处理两张图，消息 QPS 不能直接当作图片 QPS。`--event-time now` 将事件时间改为发送时刻；默认 `preserve` 完全保留原事件时间，端到端观察使用本轮发送时间而非旧事件时间。

生产端等待 broker ACK，并通过 `--max-inflight`（默认 20）限制未完成发送数。达到并发上限会反压，实际发送速度可能低于目标 QPS；没有无界任务或失败后的自动重发。Ctrl+C 会清理客户端并保存已得到结果的发送记录。

## 3. 分级增加输入

先验证 10 张真实链路，再按实际处理能力提高输入。示例为持续一分钟的目标 10 条/秒：

```bash
python performance_tools/kafka_replay.py replay \
  --input performance_tools/results/person_snapshot.jsonl \
  --duration 60 --qps 10 --wait-seconds 60 \
  --log-file logs/ALog.log \
  --metrics triton=http://127.0.0.1:8003/metrics \
  --metrics milvus=http://127.0.0.1:9091/metrics
```

`--duration` 单独使用时按时长发送；同时指定 `--count` 和 `--duration` 时，先达到的限制生效。需要的条数超过快照时循环使用该有限快照，仍为每次发送生成不同 `uId`，不会边读 Kafka 边产生新输入。有限快照保留原画面的人数、分辨率与图片大小分布；少量样本循环回放会复用对象存储缓存和已有身份，只能代表该样本集的容量。

示例指标端口沿用当前主机观察方式，实际地址取决于部署网络。容器里的 `127.0.0.1` 指向该容器；服务不在同一网络命名空间时应改为可达地址。指标获取失败只记录异常类型，不阻止有限回放。可重复添加 `--metrics NAME=URL`，记录观察窗口开始、结束的 Prometheus 指标快照；counter 可据差值计算速率，gauge 差值没有吞吐意义，发生进程重启或 counter 重置时不能直接相减。GPU 使用率请同时在 Jetson 主机采样 `tegrastats --interval 1000`；本脚本不尝试获取 root，也不把不同时刻的 GPU 值当成窗口平均值。

## 4. 读取结果与耗时

每轮在 `performance_tools/results/<run_id>/` 保存：

| 文件 | 含义 |
| --- | --- |
| `sent.jsonl` | 每条独立 `uId`、输入来源、发送时间、ACK 时间和最终 Kafka offset；失败记录异常类型 |
| `outputs.jsonl` | 五类输出 Topic 中仅属于本轮 `uId` 的记录定位和延迟；不复制向量 embedding |
| `summary.json` | 实际发送量与速度、输出量、首条输出延迟、整图向量写库后输出延迟、指标前后快照、同窗口 ALog 性能汇总 |

输出观察消费者同样不加入消费组、不提交 offset，从发送前捕获的输出 Topic 末尾开始读；有其他业务流量时，仅按本轮 UUID 匹配结果。默认观察所有五个输出 Topic；若测试环境缺少输出 Topic，可显式使用 `--no-observe` 只测试发送。输出数量为观察截止时的结果，异步写入或积压可能在窗口结束后完成，因此“没看到”不是最终失败结论，应加长 `--wait-seconds` 或继续查看日志。

ALog 默认 INFO 周期输出 `ALog performance summary: {JSON}`。使用 `--log-file` 后，脚本采集从回放开始到观察结束的新汇总，包含：

- `operations.image_decode`：注入的 `image_cpu` 工作线程中 `cv2.imdecode()` 的调用次数、错误次数、总耗时、平均和最大毫秒数，不含图片下载、名额等待或线程调度。
- `operations.image_decode_queue_wait`：从下载完成后请求解码，到工作线程实际开始解码的等待；正式入口与其他图像工作共享有界 CPU 资源。
- `operations.person_seg_unpack`：同步分割结果解包与后处理的次数、错误和耗时。
- `operations.person_crop_prepare`：工作线程内人物裁图/转正与结构质量检查耗时；不含后续视觉检查、编码、上传。
- `operations.person_crop_prepare_queue_wait`：请求人物裁图准备到工作线程实际开始的名额等待与调度耗时。
- `operations.face_identity_lock_wait` / `person_identity_lock_wait`：成功取得对应全局身份锁的等待耗时；未取得锁就取消的请求不计入。
- `operations.face_identity_lock_hold` / `person_identity_lock_hold`：每次身份解析/提交锁内处理耗时；稳定预取的远程查询在锁外，完整回退查询仍计入持锁。Face 刷新路径可产生两次 hold。锁内查询包含客户端线程等待和检索，不能只用 Milvus 服务端 Search 时长代替。

Face 的 `face_identity_lock_wait/hold` 由 `FaceIdentityService` 统一记录，覆盖独立人脸和人物内部 Face 复用/回退的身份解析持锁路径；`stages.face_identity` 也覆盖两条路径，快照短锁另列。相比旧版只计人脸提交入口的统计新增了人物路径，跨版本比较必须注明口径变化，不能把新增样本误认为锁退化。

四个资源域 `image_cpu/storage_io/vector_search/vector_write` 均提供 `operations.<name>_run` 和 `<name>_queue_wait`，分别测真实线程工作与执行前等待。旧解码、裁图分项是同一次工作的附加计时，与资源域指标有重叠，不能相加。调用方取消后仍执行的工作继续占许可并记录时长；`queues.resources` 的 inflight 表示已提交但尚未由事件循环完成回调回收的任务，含线程已结束但回调尚未执行的情况，不能当作瞬时忙线程数。

Face 预取使用下列 `operations` 指标，整图人脸与人型内部 Face 查询共用预取逻辑：

| 指标 | 计时或计数边界 |
| --- | --- |
| `face_identity_requests` | 逻辑 Face 解析开始次数；包括独立人脸提交、人物内部只读查询，以及后来失败/取消的请求 |
| `face_identity_snapshot_lock_wait/hold` | 初次预取前读取本实例图库/pending 移除版本的短身份锁等待与持有；额外刷新取版本发生在身份解析锁内 |
| `face_identity_prefetch` | 初次预取及额外刷新两种锁外远程查询的墙钟耗时，包含线程调度与 I/O 等待 |
| `face_identity_prefetch_used` | 当前解析成功采用稳定候选的次数，包括初次候选和刷新候选 |
| `face_identity_prefetch_fallback` | 一份候选在状态复核时失效的次数；包括初次候选和刷新候选，随后可能先锁外刷新，不直接等于完整锁内查询 |
| `face_identity_prefetch_skipped` | 起始时实际写线程活跃或缺少版本能力而跳过初次预取；稍后现态稳定时仍可能进行一次刷新 |
| `face_identity_prefetch_failed` | 初次或刷新预取发生普通异常的次数；取消不计入 |
| `face_identity_refresh` | 额外锁外查询的次数和耗时，每个逻辑请求最多一次；同时计入 prefetch，不能与之相加 |
| `face_identity_refresh_used` | 刷新候选通过版本复核并被采用的次数 |
| `face_identity_refresh_invalid` | 刷新候选在重新持锁复核时失效的次数 |
| `face_identity_refresh_failed` | 额外锁外查询发生普通异常的次数；取消不计入 |
| `face_identity_refresh_unavailable` | 尚未刷新但当前没有稳定版本或预取能力，决定使用完整锁内查询的次数；包含禁用预取 |
| `face_identity_refresh_exhausted` | 已尝试一次刷新后仍决定完整锁内查询的次数；可与 invalid 或 failed 同次出现 |
| `face_identity_locked_search` | 最终完整锁内查询的次数和耗时；包含随后失败/取消的查询尝试，不能当作成功数 |
| `face_identity_locked_full_pending` | 身份服务持锁期间，完整搜索调用 pending 查询的墙钟时间，包括可能的内部锁等待 |
| `face_identity_locked_full_milvus` | 身份服务持锁期间，完整搜索调用 Milvus 客户端的墙钟时间，包括接纳、批处理、执行和调度等待，不是服务端执行时间 |
| `face_identity_locked_full_resolve` | 身份服务持锁期间，完整搜索合并候选并解析的墙钟时间 |
| `face_identity_locked_prefetch_pending` | 身份服务持锁复核预取候选时的 pending 查询墙钟时间；包含读取后候选仍失效的尝试 |
| `face_identity_locked_prefetch_resolve` | 身份服务持锁且预取候选通过复核时，合并候选并解析的墙钟时间 |
| `face_identity_locked_query_admission` | 锁内完整人脸向量查询从开始等待协调器容量到接纳的时间 |
| `face_identity_locked_query_batch` | 接纳后入队至被选入批次的时间，包括凑批、协调器批次容量等待和调度 |
| `face_identity_locked_query_executor` | 被选批到工作线程开始 SDK 调用的时间，包括协调器任务调度、共享搜索池许可等待和线程调度 |
| `face_identity_locked_query_sdk` | 工作线程执行 `client.search` 的墙钟时间，包括 SDK、gRPC 和服务端等待，不是服务端纯执行时间 |
| `face_identity_locked_query_resume` | SDK 工作线程返回到原调用协程恢复的时间，包括跨线程通知、批次结果校验分发和事件循环调度 |
| `face_milvus_search` | Milvus 客户端查询耗时，包含初次预取、刷新与锁内回退；不是服务端计算时间 |
| `face_pending_search` | 当前进程 pending 查询耗时，包含适用的快照锁等待 |
| `face_match_resolve` | 合并候选后执行已有身份与灰区独立样本解析的耗时 |

`requests`、`prefetch_used/fallback/skipped/failed` 和 `refresh_used/invalid/failed/unavailable/exhausted` 是零耗时事件，只取 `count`。一次逻辑解析可能有两份候选、两次身份锁等待/持有；`used / (used + fallback)` 仅描述复核正常返回“通过或失效”结论的候选尝试，复核抛错或取消的尝试不进入分母，不能当作逻辑请求的使用率。不要用 used/fallback/skipped 之和或 Milvus 次数减 fallback 推算新版本逻辑请求或锁内查询；应读取 requests 和 locked_search，并使用包含开始与结束的完整观察窗口，避免周期桶边界错位。

上述 `face_identity_locked_full_*` 与 `face_identity_locked_prefetch_*` 仅覆盖身份服务持锁的当前协程调用 `FaceSearchService` 的分项；锁外直接调用和自定义搜索回调不计入。分项记录墙钟耗时，失败或取消也会记录并计入 errors；它们与 `face_identity_lock_hold`、`face_identity_locked_search` 及原有对应查询指标有嵌套，不能相加。各操作在结束时进入周期桶，跨窗口、异常及不同路径会使次数与耗时无法逐桶严格对应，也不能将墙钟时间直接视为 CPU 或磁盘时间。

`face_identity_locked_query_*` 只统计身份服务持锁协程通过 `FaceSearchService` 完整查询发起的每条向量请求；同一兼容批次中的锁外请求不计入，多条锁内请求共享批次时各自记录相同的线程/SDK 阶段，因此计数是查询数、不是独立 RPC 数，累计时间不可解释为服务端资源占用。请求在提交后取消时，只保留协调器在取消前已记录的分项；即使 SDK 已经返回，若协调器尚未处理结果就发生取消，线程和 SDK 耗时也可能不计。真实工作完成前不会释放容量；SDK 失败记入对应 SDK 和已恢复阶段的 errors。各阶段在完成时计入周期桶，且不包含查询参数构建等少量开销，不宜与 `face_identity_locked_full_milvus` 逐桶严格相等。现有服务端指标为聚合口径，未实现逐请求配对。

没有有效候选且当前状态稳定时，服务最多额外刷新一次；当前仍有 active writer、能力缺失或刷新后仍不可用时才使用原完整锁内查询，不等待 writer 空闲。`FACE_SEARCH_PREFETCH_ENABLED=false` 不产生预取/刷新 RPC，仍记录 requests、refresh_unavailable 和 locked_search。锁外预取普通异常不当作正常未命中；锁内候选复核或刷新状态读取本身抛错，则直接作为终态查询失败，不再刷新或重查。最终查询异常禁止新建身份；取消直接传播，业务决策与 pending 登记异常不触发刷新。总锁占用须合计初次快照及所有身份解析锁 hold 的累计值，不能只比较每次平均值。候选版本只覆盖当前实例，不提供跨进程事务。

人型身份指标：`person_identity_requests` 计有效 embedding 的逻辑请求开始，包含后来失败/取消的请求；`person_identity_locked_search` 直接计最终完整锁内查询次数和耗时。`person_identity_snapshot_lock_wait/hold` 为前置快照锁等待/持有，`person_identity_prefetch` 包含初次及一次额外锁外查询，`person_identity_refresh` 单列刷新，两者不能相加。`person_identity_prefetch_used/fallback/skipped/failed` 仅取 count，表示稳定复用、候选复核失效、起始状态不可预取、锁外查询普通异常。fallback 后可能刷新，skipped 后若写线程已经结束也可刷新，因此不能用这些计数之和推算请求数或锁内查询数。初次预取异常传播，刷新普通异常进入原完整回退，取消直接传播且不计 failed。

`person_identity_refresh_used/invalid/failed/unavailable/exhausted` 分别计刷新候选通过、失效、普通异常、当前无刷新条件和刷新后仍回退。一次请求最多经历两次提交锁；总锁持有时间须相加 snapshot 与 commit hold 的累计值。`person_identity_search` 按持锁搜索段计数，需要按逻辑请求计算均值时，以 requests 为分母。`person_reid_milvus_search` 包含初次、刷新及回退；锁内查询率使用 locked_search / requests，不与旧版 fallback 比例直接混用。版本校验只覆盖当前进程，回退仍使用原有查询顺序，不是跨库事务。

Face 查询默认按批大小 4、窗口 1 ms、容量 128 调度，所有异步查询共用此路径。`vector_query_admission_wait` 和 `vector_query_batch_wait` 按向量计前段等待，`vector_query_rpc_run` 按批次计 SDK 执行，与 `vector_search_run` 重叠；`queues.resources.vector_queries` 的 nq、RPC 和 batch 直方图用于确认实际合批。搜索池排队下降可能只是等待转移到接纳或凑批阶段，不能单独说明提速。SBS 继续保持按身型分支查询的过滤和召回额度，未并入 Face 微批。

SBS 检索的 `sbs_search_thread_queue` 统计请求 `vector_search` 资源至线程开跑的等待，`sbs_search_thread_run` 统计线程内客户端调用，`sbs_search_loop_resume` 统计线程结束到事件循环计时回调执行的延迟。最后一项是回调调度观测，不是精确的调用方 `await` 恢复时间；每个人物的多类 gallery 检索仍并行，不能将三次调用耗时相加当作持锁时间。

SBS 各检索分支使用显式注入的 `vector_search` 执行器，再通过一次 `gather` 等待。各分支独立复制 context，结果仍按提交顺序返回。关联计时输出下列固定名称指标，不保存身份、查询内容或逐请求日志。下表名称均带 `sbs_search_` 前缀：

| 指标后缀 | 计时边界 |
| --- | --- |
| `future_callback` | 每个未取消 Future 的业务函数结束到其完成回调执行；区别于原统计回调的 `loop_resume`，也不是逐分支 `await` 恢复 |
| `group_wait` | 从创建检索组到 `gather` 返回或抛错，含分支提交与调度，不含后续候选合并 |
| `future_group_submit_span` | 各分支开始提交的最早与最晚时间差；仅反映提交错峰，不加入关键路径总和 |
| `future_critical_dispatch` | 检索组开始到关键分支开始提交 |
| `future_critical_thread_queue` | 关键分支开始提交到业务线程函数开始，含资源名额、执行器与 Python 调度等待 |
| `future_critical_thread_run` | 关键分支业务线程函数的完整执行，包含 SDK/RPC 和候选构造 |
| `future_critical_callback` | 关键分支业务线程函数结束到其 Future 完成回调执行 |
| `future_group_join` | 关键分支 Future 完成回调到 helper 在 `gather` 恢复后采样的时刻，含调度与分支统计开销，不含本组最终聚合写入自身的耗时 |

关键分支选取本组 Future 完成回调最后执行的一路。成功完整组的五段 `future_critical_*` / `future_group_join` 之和等于该组 `group_wait`；它们均为墙钟时间，不是 CPU 忙碌时间。失败或取消组只记录带错误标记的 `group_wait`，不构造完整关键路径；已结束且未取消的 Future 仍可有 `future_callback`。取消排队中的 Future 可以阻止执行，取消后继续运行的线程仍沿用原线程统计。跨周期聚合、失败样本或其他业务输入会使各指标样本集合不同，比较前应检查次数与错误数。各身型分支的召回数量、异常传播和结果顺序不变；当前候选检索使用 Strong，身份预取与回退边界见上文。与旧的逐分支 `await` 实现对照时，应比较同义的 `group_wait` 和身份锁持有时间，不能把 Future 回调延迟与实际 `await` 恢复延迟直接当作同一指标。

验证 Triton 跨请求 GPU 合批时，同步保存服务端以下计数的窗口增量。它们由 `orin_gpu` 的 `yolo_orin_metrics.py` 提供，按 model/version/instance 标记；Pose/Seg 的合批在一次 `execute(requests)` 内发生，总 B≤4，不额外等待下一次 execute。

| 服务端指标 | 含义 |
| --- | --- |
| `yolo_orin_trt_calls_total{batch_size}` | `execute_async_v3` 边界的真实 TensorRT enqueue 尝试，批大小分别为 1..4；不是成功完成量 |
| `yolo_orin_trt_failures_total` | enqueue 失败次数，与尝试数一起检查 |
| `yolo_orin_group_failures_total` | 已校验执行组的失败次数，可能发生在组包、推理或响应拆分阶段 |
| `yolo_orin_copy_bytes_total{direction}` | `host_pack`、`host_split`、`input_h2d`、`output_d2h` 的逻辑 Tensor 复制量，不是实测内存总线流量 |
| `yolo_orin_host_copy_seconds_total{direction}` | 主机组包和拆分的墙钟耗时，direction 为 `host_pack` 或 `host_split` |

以实际 enqueue 批大小分布核对 GPU 是否收到更大的批，再比较新增 host pack/split 开销、错误数和完整业务排空；客户端 batch 次数、服务端 request 数或 GPU 利用率单项上升不能作为吞吐提升结论。各模型的输入 shape 分布也要固定，跨请求合批仅合并同形状图片，额外主机组包上限 96 MiB。模块与响应契约见[架构说明](../docs/代码架构与处理流程.md#82-单次推理请求的内部流程)。

开启 `TRITON_SYSTEM_SHM_SLOTS` 后，可从 `operations` 读取以下指标，部署条件见[项目 README](../README.md)。

| 指标 | 准确含义 |
| --- | --- |
| `triton_shm_copy_queue` | 申请复制线程名额到工作线程开跑的墙钟等待，包含有界池背压和线程调度 |
| `triton_shm_copy_run` | 工作线程内公开 SDK 的内存复制耗时，不含 RPC；已提交后取消的任务仍会记录 |
| `triton_shm_used` | 调用 SDK 前采用 SHM 的推理尝试次数，可能随后失败或取消，不表示成功数 |
| `triton_shm_fallback` | 启用 SHM 后，准备阶段未使用槽的调用次数；隔离槽后的原传输重试每次另计一次 |
| `triton_shm_poisoned` | 首次因推理尝试异常或取消而永久隔离的槽数，同一槽不重复计数，不补建 |

后三项使用 `record(name, 0)` 记事件，只读取 `count`，不把 `avg_ms` / `max_ms` 当作耗时。`fallback` 包含其他契约、BYTES、小于 1 MiB、超过单槽、无空槽或槽已隔离等情况，甚至可能包含随后本地校验失败而未发送的调用；它不能用来计算槽不足率。重试会增加事件计数，不能直接把 `used + fallback` 当作 RPC 总数或图片数。SHM 关闭时不会产生这些事件。

逐请求 DEBUG 明细里的 `serialize_ms` 是首次输入准备的墙钟时长：使用 SHM 时包含线程名额等待、内存复制、事件循环恢复和输入对象构造；未显式启动 transport 时还可能含首次 SHM 初始化。它不等于 CPU 忙碌时间或事件循环阻塞时间，不能与 `copy_queue + copy_run` 直接相加，也不包含 SDK 在 `client.infer()` 内的全部 protobuf/gRPC 构造。隔离槽后的原传输重试会在 RPC 路径重新构造输入，这部分计入 `rpc_ms`。周期性能汇总目前没有按模型单独聚合 `serialize_ms`，常规比较优先看上述复制指标、`event_loop_lag` 和完整图片完成速度。

DEBUG 的 `request_bytes` 仍统计输入 Tensor 的逻辑 payload，SHM 数组也按原字节数计入；它不是 gRPC 报文大小或实测网络流量，不能用它判断 SHM 节省了多少网络复制。

模板保持 0 槽和 private IPC，支持跨机器部署；本机同机配置为 16 槽 × 32 MiB，即固定 512 MiB，并使用两端 host IPC。多进程按全局预算分配槽位，共享复制线程总计最多 4 个。最多 16 个大图 RPC 同时占槽，超过时即时使用普通传输，不能据此假设覆盖全部执行器并发或达到目标 QPS。验证时保持同一快照、发送节奏、业务并发与预热条件，结合完整图片完成速率、复制等待、隔离计数、Kafka lag、Milvus 输出及实际内存判断；单模型 RPC 微基准只说明该模型的传输收益。

- `operations.event_loop_lag`：事件循环定时探针迟到量，可以与上述同步操作和 `batch_wait_ms` 对照。
- `inference_by_model`：各模型 batch 次数、样本数及组批、执行器等待、RPC 耗时。
- `images_completed_by_topic`：图片业务流程完成计数，主图和 pano 独立计图；`messages_completed_by_topic` 是消息返回量，可能包含正常跳过的输入。

组包的 `pack_ms` 和新增 `pack_queue_wait_ms` 属于逐请求 DEBUG 指标，目前不在 `inference_by_model` 周期汇总中。前者只计真实组包执行，后者单列 CPU 等待并加入 `queue_total_ms`；CPU 域自身仍提供资源聚合计时。预计复制至少 1 MiB 的非连续 numeric 或多样本 numeric 堆叠每批借用 `image_cpu` 一次；连续单图和 encoded bytes 包装保持同步轻量路径。输出校验/解包不改变执行位置。

这些周期计数包含同一进程、同一窗口的全部业务输入，不能在有其他输入时都归因到本轮；首尾汇总周期也可能跨出回放窗口。精确定位某条消息可临时使用 DEBUG 的 `Successfully processed message ... uId=...` 日志。日志轮转覆盖窗口时，本工具只读取当前文件，需同时查看轮转文件。

需要精确统计本轮完整图片业务完成时间时，在 `replay` 命令显式添加 `--trace-completion`，并继续提供 `--log-file` 采集周期汇总。默认仍生成 UUID5；该选项仅用于本测试链路，将 `uId` 改成 `alogtest-v1:<32位UUID4 run_id>:<sequence>`，最长 50 字节，是合法非空字符串，但不是 UUID，使用前需确认外部输出消费者能接受。它不增加 payload 字段，不改变图片字段、时间或存储结构。每轮最多 65536 条，长时回放超限会报错并保留已发送结果。发送采用绝对 monotonic 时间计划，避免每次调度迟到累积降低输入速度；落后时仍受 `--max-inflight` 限制。

`performance` schema 2 的 `replay_runs` 只识别上述合法显式标记，不为普通生产 UUID 增加 run 状态。每进程最多保留 16 个 run，每个 run 用主图和 pano 各 8 KiB 的位图去重，1 小时无完成事件后过期。摘要仍每约 5 秒输出一次，包含累计完成数及实际首末完成时间，不逐帧输出 INFO 或 UID。完成点在整图、人脸、人型均返回之后；异步 Milvus/Kafka 缓冲可能仍未排空。`generation` 与 `replay_run_evictions` 用于识别容量淘汰，重建后的 run 不直接拼接累计数。

**`picture_vectors_persisted_and_emitted` 表示整图向量 Milvus 写入成功后，相应 Kafka 向量记录已经被本工具观察到；这不是整帧完整处理完成数。** 整图向量可能在该帧人脸、人型仍在处理时先落库；人物和人脸又可产生多行向量，不能把总写入行数当成图片数。完整容量判断应同时满足目标图片完成速度、Kafka lag 与内部队列不持续增长、延迟稳定，并单独核对异步写库及输出。

`results/` 已被 Git 忽略。快照包含真实设备 SN、图片路径和业务消息，应仅留在本地运行目录，勿提交到仓库。

需要比较优化前后时，可离线分析两轮结果，不必重新连接 Kafka：

```bash
python performance_tools/analyze_run.py \
  performance_tools/results/<before_run_id> performance_tools/results/<after_run_id> \
  --output performance_tools/results/qps_comparison.json
```

分析器分别报告 broker ACK 输入速度、本轮 UID 的整图向量写库后输出速度、同 Topic 的业务图片周期完成量，以及显式追踪模式的 `business_images_by_run`。后者以 `sent.jsonl` 的真实首次发送时刻与 run 的实际末次主图完成时刻计算 `first_send_to_last_completion_seconds` 和 `observed_images_per_second`，不使用汇总桶的首尾时刻；`completed_count` 是去重后的主图完成数，pano 另列。仅在 ACK 序号连续完整、主图全部完成且未启用 pano 时，将 `full_pipeline_completion_verified_for_run` 置为 `true` 并给出精确业务排空时间；多进程、追踪状态重建、计数回退或时钟回退均保守标记为未验证。周期快照中完全没有本轮状态时，精确完成数为 `null`，不把缺少观测当成零完成。

`business_images_by_topic.periods` 保留每个实际约 5 秒周期的完成数与速率；`peak_fully_contained_period` 排除跨出回放窗口的周期，`active_period_span_images_per_second` 从第一个有图片完成的周期开始到最后一个有图片完成的周期结束计时，包含中间空闲周期。它们仍包含同进程同 Topic 的其他业务，不代表本轮精确 QPS；短样本的活跃周期均速也不能代替持续容量。

旧 UUID 模式及 Topic 聚合的 `run_business_drain_seconds` 保持 `null`；显式追踪通过全部主图验证后，`business_images_by_run.run_business_drain_seconds` 才表示最后 ACK 至最后主图业务完成的精确间隔。`last_observed_topic_image_period_after_last_ack_seconds` 仅给出最后一个有图片完成的 Topic 周期相对最后 ACK 的时间范围；整图向量的 `after_last_ack_seconds` 仅属于向量持久化输出口径。异步写入、Kafka lag 和内部队列仍需单独观察。可重复传 `--log-file logs/ALog.log` 或轮转日志补齐周期，分析器按回放时间过滤并去重，不会将跨边界周期按时间比例拆分。

### 结合阶段队列日志解释等待

同时保留本轮 `logs/pipeline_stats.jsonl` 及相关轮转文件，作为独立阶段证据。它每约 5 秒记录一次，字段与查看方法见[流水线队列统计与写出隔离](../docs/流水线队列统计与写出隔离.md)。不要把低频采样峰值当作真实峰值，或把快照中的全部业务归因于本轮；比较时核对 `pid/session_id`、完整覆盖时间和是否存在其他输入。字段缺失表示没有该项观测，不能直接填零。

| 路径 | 压测时如何解释 |
| --- | --- |
| `queues.input.queued_messages` | 尚未由领取 worker 取出的本地消息数；不包括 broker lag |
| `queues.messages.capacity_messages/inflight_messages` | 业务消息容量及已接纳未结束的消息数；一条消息可能包含主图和全景图 |
| `queues.messages.admission_waiters`、`stages.message_admission` | 已领取但等待业务容量的消息数与等待时长；接纳返回不是业务完成 |
| `queues.messages.closed` | 调度器是否已停止接纳；正常退出仍须确认在途归零 |
| `queues.frames.capacity_frames/inflight_frames/waiting_frames` | 下载前的图片容量与在途/等待数；主图和全景图独立计图 |
| `queues.frames.reserved_bytes/observed_image_bytes` | 固定预留与当前 lease 的原图像素加压缩字节观测，均不是 RSS 硬上限 |
| `queues.resources.<name>` | 四类资源的等待与真实在途工作；调用方取消不提前释放仍运行的线程许可 |
| `queues.resources.vector_queries` | Face 查询统一调度器的接纳/排队向量、在途批次、完成 RPC 与 nq、batch 直方图；capacity 包含排队和在途，batch size=1 时仍存在 |
| `stages.face_ordered_commit/person_ordered_commit` | 准备真正结束后至实际开始提交的等待；不包含推理尚未完成的时间 |
| `stages.face_commit/person_commit` | 正在提交的目标及正常完成目标/s，包含身份查询、锁等待与记录入队；不是新增身份速率或纯计算时长 |
| `stages.face_identity_snapshot`、`stages.face_search_prefetch` | 预取短锁等待/持有，以及锁外远程查询活动；用于区分等待转移与锁占用减少 |
| `stages.person_prepare` | 人物准备整体任务活动，仍需等待视觉与 Face 两分支汇合 |
| `stages.person_visual_prepare`、`stages.person_face_match` | 视觉和 Face 分别受独立名额限制；Face 等待不占人物视觉许可 |
| `queues.output.total_outstanding` 及逐队列 ACK/重试 | 业务处理结束后的持久化/发布积压；需要与业务完成 QPS 一起对照 |

记录有效配置时同时保存 `ALOG_WORKER_COUNT`、消息/图片接纳、四类线程池容量、两类 `FRAME_CONCURRENCY/GLOBAL_CONCURRENCY/PREPARE_LOOKAHEAD`、`PERSON_FACE_CONCURRENCY`、`FACE_SEARCH_PREFETCH_ENABLED` 和帧字节预算。消息容量为 0 时沿用 worker 数，图片容量为 0 时沿用有效消息容量；`PERSON_FACE_CONCURRENCY=0` 继承人物全局容量但使用独立许可。每帧同时准备上限为 N，未提交总容量为 N+M，包含正在提交的目标；M=0 保持原 N 项窗口，同帧串行需 N=1、M=0。

`FRAME_MEMORY_BUDGET_BYTES=0` 默认关闭估算字节预算；启用时按 `FRAME_RESERVATION_BYTES` 固定预留，原图超出预留只记录，不在持有图片后等待扩容。`observed_image_bytes` 不计 crop、mask、Tensor、SDK 缓冲或取消后仍在线程中使用的图片，不能代替 RSS。迁移配置后核对 `queues.resources` 的有效容量；图片解码和准备只使用显式注入的 `image_executor`，独立朝向评估器也负责其创建和关闭。`IMAGE_DECODE_WORKERS`、`PERSON_PREPARATION_WORKERS`、`IO_THREAD_WORKERS` 及两套惰性图片池已移除，存储容量必须由正值 `STORAGE_IO_WORKERS` 指定。

提高前瞻容量后，就绪等待增多可能来自准备供给更快；提高消息容量后，身份锁、共享资源或输出等待也可能增多。只有同输入下完整业务完成速率提高、输出能够排空且时延/内存可接受，才能认定整体收益。单个阶段变快、锁平均等待下降或 GPU 利用率上升都不足以单独确认容量提高。正常停机 `queue.join()` 等待真实消息任务结束，再关闭调度器与缓冲输出；输入 offset 的本地入队后提交时机、进程内缓冲和新身份持久化门控均未因本轮隔离改变。consumer/dispatcher 异常会清理并传播，不应将故障退出窗口与正常排空窗口混为一轮稳定样本。

## 执行器调度微基准

`benchmark_inference_scheduler.py` 只构造推理执行器，用 `sleep(0)` 模拟异步操作让出事件循环，不连接 Triton、Kafka、存储或 Milvus。修改执行器前先把旧源码复制到 ignored 产物目录，然后比较相同配置：

```bash
cp core/infer/executor.py performance_tools/results/executor-before.py
# 修改执行器后运行；与真实回放错开，避免相互争用CPU。
python performance_tools/benchmark_inference_scheduler.py \
  --baseline-executor performance_tools/results/executor-before.py \
  --loop uvloop --concurrency 80 --operations 2305 --producers 1,8,80 --repeats 5 \
  > performance_tools/results/executor-scheduling.json
```

输出保存两份源码 SHA256、每轮完成数、实际并发峰值，以及交替运行的 wall/process CPU 时间。计时运行不插桩；另跑一轮统计 `Condition.wait` 返回次数、没有工作却被唤醒的次数和广播次数，排除启动/关闭广播。`--loop asyncio` 可在没有 uvloop 的环境使用。该实验用于判断调度通知的成本，不包含真实 RPC 或业务 CPU 工作，不能由微基准加速倍数推导完整图片吞吐，也不能把 profile 的协程累计时间当作 CPU 耗时。

## 验证工具本身

```bash
python -m unittest performance_tools.test_kafka_replay
python -m unittest performance_tools.test_analyze_run
python -m unittest tests.test_triton_executor_notifications
python performance_tools/kafka_replay.py --help
```

单测使用内存消息，不连接 Kafka、不写 Milvus。它检查冻结 offset 上界、无效输入过滤、原始字段保留、独立 UUID 关联、pano 计图及限量参数校验；不能替代真实链路压测。
