# 配置与运行说明

## 配置

配置由 [core/config.py](../core/config.py) 统一加载：**运行时环境变量优先于项目根目录 `.env`，未提供时使用代码默认值**。必填值缺失或为空会导致启动失败。

### 算法参数档案

默认算法参数来自 [config/algorithm.toml](../config/algorithm.toml)，Docker 镜像内置该档案，Compose 不挂载源码。

如需覆盖，在 `.env` 中设置可选的 `ALGORITHM_CONFIG_PATH`；相对路径从 `/app` 解析，绝对路径可用于挂载的文件。例如在 `docker-compose.yaml` 的 `visionflow` 服务中增加如下字段：

```yaml
environment:
  ALGORITHM_CONFIG_PATH: /run/visionflow/algorithm.toml
volumes:
  - visionflow-logs:/app/logs
  - ./algorithm.toml:/run/visionflow/algorithm.toml:ro
```

上述片段是服务字段示例，保留了现有日志卷。修改档案后需重启 VisionFlow，运行中的进程不会热加载。启动时会严格校验所选 TOML；文件缺失、语法错误、未知参数或不合法的取值都会使启动失败，不会静默退回默认档案。参数的单位、边界及含义见档案中的中文行内注释；无需为每个阈值增加环境变量。

人脸图库每个 Face ID 的向量上限由 `face.gallery.max_vectors_per_id` 设置，默认 `20`，范围 `1–1000`。从旧版本升级时，请把 `.env` 中的 `FACE_MAX_VECTORS_PER_ID` 移到选用的 TOML 档案；旧变量仍存在会使启动失败，以免忽略原值后误删样本。调低上限后，该身份下次写入时会按质量分和采集时间清理超额向量。

### 连接信息

| 服务 | 必填变量 |
| --- | --- |
| PostgreSQL | `PGSQL_HOST`、`PGSQL_PORT`、`PGSQL_DB`、`PGSQL_USER`、`PGSQL_PSWD` |
| Milvus | `MILVUS_URI`、`MILVUS_TOKEN` |
| Kafka | `KAFKA_BROKER_ADDR`、`KAFKA_BROKER_PORT`、`KAFKA_GROUP_ID` |
| Triton | `TRITON_SERVER_URL`，填写 VisionFlow 可达的 gRPC 地址，例如 `triton-host:8001` |
| OSS（`STORAGE_BACKEND=oss`，默认） | `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`、`OSS_ENDPOINT`、`OSS_BUCKET_NAME` |
| MinIO（`STORAGE_BACKEND=minio`） | `MINIO_ENDPOINT`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`、`MINIO_BUCKET_NAME`；`MINIO_SECURE` 默认 `false` |

`KAFKA_SECURITY_PROTOCOL` 的代码默认值为 `SASL_PLAINTEXT`，模板值为 `PLAINTEXT`。使用 SASL 时还必须填写 `KAFKA_USERNAME`、`KAFKA_PASSWORD`，机制由 `KAFKA_SASL_MECHANISM` 指定，默认为 `PLAIN`。`KAFKA_AUTO_OFFSET_RESET` 默认为 `earliest`。

PostgreSQL 参数保留必填校验；生产入口不创建 PostgreSQL 会话或执行其连通性检查，识别查询也不使用 PostgreSQL。

### 图片对象存储

`STORAGE_BACKEND` 只接受 `oss` 或 `minio`，未配置时使用 `oss`，兼容现有 OSS 部署。启动时只要求所选后端的连接参数；原图下载、人脸裁图和人物裁图上传统一使用该后端。修改后须重启 VisionFlow，不支持热切换。

切换到 MinIO 时，在 `.env` 中设置：

```dotenv
STORAGE_BACKEND=minio
MINIO_ENDPOINT=minio-host:9000
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_BUCKET_NAME=visionflow-images
MINIO_SECURE=false
```

将 `minio-host` 替换为 VisionFlow 可达的宿主机 IP 或域名，并填写凭据。`MINIO_ENDPOINT` 使用 `host:port`，不带 `http://` 或 `https://`；`MINIO_SECURE=true` 使用 HTTPS，`false` 使用 HTTP。连接对象存储 API 端口 `9000`，不要使用控制台端口 `9001` 或服务器磁盘目录。当前 Compose 使用宿主机网络；跨机器部署不能使用 `localhost` 或 MinIO 容器名。同机地址也需确认从 VisionFlow 运行环境可达。

可复用已有 Milvus 的 MinIO 服务，但业务图片建议使用独立 bucket，如 `visionflow-images`，不要混入 Milvus 内部的 `a-bucket` / `files` 数据。VisionFlow 不自动创建 bucket；运行前需由存储管理员准备 bucket，并为所填凭据授权读取原图、写入裁图。

输入推荐使用裸 object key；MinIO 完整 URL 仅接受与配置 endpoint、bucket 匹配的路径式地址，详见 [Kafka 输入契约](kafka_topic.md#0-输入-topic)。Kafka 图片字段和 Milvus 图片元数据继续保存 object key，不生成签名 URL。切换前须同步上游图片上传服务和下游图片读取服务；开关不会迁移历史 OSS 图片，也不会自动回退到旧后端或双写。

### 并发与组批

下表的代码默认值与根目录 [.env.example](../.env.example) 保持一致，采用 `baseline` 资源档位：一个 owner 和四个视觉进程、统一图片 CPU 池、Face 查询合批及 Triton B4 组批。其他环境仍需同输入实测；环境变量可覆盖这些值，修改后重启服务。

| 变量 | 代码与模板默认值 | 含义 |
| --- | ---: | --- |
| `ALOG_WORKER_COUNT` | 32 | 从本地队列领取并提交消息的 worker 数；接纳后由独立消息任务持有整条消息 |
| `VISUAL_WORKER_PROCESSES` | 4 | 默认四个视觉子进程；0 是受限环境可用的单进程模式，1..4 使用单 owner 与视觉进程分离 |
| `MULTIPROCESS_IPC_MAX_REQUESTS` | 128 | 每个私有 IPC 连接的请求数量额度；普通请求、帧准入和回复分别预留 |
| `MULTIPROCESS_IPC_MAX_BYTES` | 67108864 | 每个 IPC 额度组最多 64 MiB 编码消息，不是整个进程的 RSS 上限 |
| `MULTIPROCESS_IPC_MAX_PAYLOAD_BYTES` | 4194304 | 单个 IPC 消息编码上限 4 MiB，不得大于对应字节额度 |
| `MULTIPROCESS_START_TIMEOUT_SECONDS` | 90 | 多进程启动等待超时，须为有限正数 |
| `MULTIPROCESS_STOP_TIMEOUT_SECONDS` | 60 | 多进程停机等待超时，须为有限正数 |
| `ALOG_MAX_INFLIGHT_MESSAGES` | 100 | 已接纳但业务尚未结束的消息数上限；0 沿用 worker 数，正整数独立设定，包含同一消息的主图和全景图 |
| `OUTPUT_BACKLOG_HIGH_WATERMARK` | 2000 | 输出 queued + inflight 达到此值时，暂停领取新消息；在途帧继续完成，非内存硬上限 |
| `OUTPUT_BACKLOG_LOW_WATERMARK` | 1000 | 输出降至此值恢复领取，须小于高水位 |
| `ALOG_MAX_INFLIGHT_FRAMES` | 100 | 下载前接纳的图片数上限；0 沿用有效消息容量，主图和全景图各自计图 |
| `FRAME_MEMORY_BUDGET_BYTES` | 0 | 固定预留的估算字节预算，0 关闭；非 RSS 硬上限 |
| `FRAME_RESERVATION_BYTES` | 67108864 | 每张在途图片预留 64 MiB；启用字节预算时至少容纳一份预留 |
| `IMAGE_CPU_WORKERS` | 4 | 原图解码、人物 Pose 绑定/朝向/结构质量、裁图/视觉质量、编码与向量归一化的有界 CPU 执行容量 |
| `STORAGE_IO_WORKERS` | 64 | 对象存储下载/上传的显式有界线程容量，必须为正，不继承默认线程池 |
| `STORAGE_DOWNLOAD_RESERVED_WORKERS` | 4 | 同一存储池内下载保底容量；上传并发至多为有效存储容量减去该值，0 保持原共享准入 |
| `MILVUS_SEARCH_WORKERS` | 16 | Face/SBS Milvus 查询的有界执行容量 |
| `MILVUS_WRITE_WORKERS` | 4 | Milvus 写入与图库更新的有界执行容量 |
| `MILVUS_QUERY_BATCH_SIZE` | 4 | Face 查询跨请求批大小上限，范围 1..32；1 仍经过同一有界查询调度器 |
| `MILVUS_QUERY_WAIT_MS` | 1 | 查询凑批窗口，有限值 0..10 ms；不包含下游搜索池等待 |
| `MILVUS_QUERY_CAPACITY` | 128 | 查询调度器已接纳排队与真实在途向量数总上限，至少等于 batch size |
| `TRITON_SYSTEM_SHM_SLOTS` | 0 | 同机 Triton 系统共享内存输入槽数，0 关闭，最多 64 |
| `TRITON_SYSTEM_SHM_SLOT_BYTES` | 33554432 | 单槽字节数，至少 1 MiB；全部槽总量最多 512 MiB |
| `FACE_FRAME_CONCURRENCY` | 30 | 每帧同时准备人脸的上限 N |
| `FACE_GLOBAL_CONCURRENCY` | 100 | 所有 worker 共享的人脸准备与推理上限；进入身份提交前释放名额 |
| `FACE_PREPARE_LOOKAHEAD` | 2 | 人脸额外前瞻容量 M；每帧准备中、等待提交和提交中合计最多 N+M 项 |
| `FACE_SEARCH_PREFETCH_ENABLED` | true | 允许锁外预取 Face 候选及最多一次锁外刷新；最终仍持锁复核版本、最新 pending 和身份规则 |
| `PERSON_FRAME_CONCURRENCY` | 30 | 每帧同时准备人物的上限 N |
| `PERSON_GLOBAL_CONCURRENCY` | 100 | 所有帧共享的人物视觉准备与 embedding 推理容量，不占用人物 Face 解析容量 |
| `PERSON_FACE_CONCURRENCY` | 0 | 人物内部 Face 解析的独立容量；0 沿用 `PERSON_GLOBAL_CONCURRENCY` |
| `PERSON_PREPARE_LOOKAHEAD` | 2 | 人物额外前瞻容量 M；每帧未提交总量上限为 N+M |
| `TRITON_BATCH_TARGET_DEFAULT` | 4 | 每个模型的目标 batch；未覆盖的远程模型须实际支持 B4 |
| `TRITON_BATCH_WAIT_MS_DEFAULT` | 2 | 客户端最大凑批等待，毫秒；不包含后续执行器或 RPC 等待 |
| `TRITON_EXECUTOR_CONCURRENCY` | 80 | 全模型共享的远程 batch RPC 并发上限 |
| `TRITON_CLIENT_POOL_SIZE` | 100 | gRPC 客户端池大小 |
| `TRITON_EXECUTOR_QUEUE_CAPACITY` | 256 | 已形成 batch 的等待队列容量 |
| `TRITON_QUEUE_TIMEOUT_SECONDS` | 60 | 从提交请求起的排队期限，含凑批和执行器等待 |
| `TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS` | 5 | 获取 gRPC client 的超时 |
| `TRITON_INFER_TIMEOUT_SECONDS` | 30 | 单次远程推理超时 |
| `TRITON_CONFIG_TIMEOUT_SECONDS` | 5 | 单次模型发现请求超时 |
| `TRITON_TRANSIENT_RETRIES` | 0 | 服务不可用错误的额外重试次数，仅允许 0 或 1 |

- 消息接纳和整条消息的执行由不同任务负责，`ALOG_MAX_INFLIGHT_MESSAGES` 独立限制已接纳任务数；0 保持原 worker 数作为执行上限。该数不含本地队列及领取后等待接纳的消息，也不是图片数量或字节内存硬上限。一个消息任务覆盖主图与全景图，真实结束后才调用本地队列 `task_done()`。
- 人脸与人物各自使用逐帧的有序调度器：同时准备不超过 N 项，未提交总量不超过 N+M 项，包含准备、等待和正在提交的结果。准备完成即释放执行名额，可在提交等待时继续准备后续目标；提交仍按原检测顺序进行，人型等待本帧人脸完成。M=0 保持原 N 项窗口；同帧串行需设置对应的 `FRAME_CONCURRENCY=1` 和 `PREPARE_LOOKAHEAD=0`。容量是每帧上限，全进程占用仍随并行帧数与图片大小增加。
- 后序准备失败时停止接纳更多目标，已接纳前序按顺序提交，到失败项再抛出异常并回收后序任务。提交失败或调用方取消时回收剩余任务；已提交结果不回滚。
- Face 查询先用短锁取得本实例版本，再在锁外查询 Milvus；进入身份锁后读取最新 pending 并复核版本。没有有效候选时，若当前可取得稳定版本且尚未刷新，则释放锁额外查询一次，再重新持锁复核；当前有真实写线程活动、缺少版本能力或一次刷新仍不可用时，使用原完整锁内查询。最终身份决策至 pending 登记始终持锁，不等待 writer 空闲或无限重试。关闭 `FACE_SEARCH_PREFETCH_ENABLED` 禁用初次预取和刷新，不改变质量门槛、灰区双样本规则、pending 登记和新身份持久化门控。人型 crop 内 Face 复用/回退也使用同一 `FaceIdentityService`；整体任务仍计入 `person_prepare`，但它等待 Face 解析时不占 `person_visual_prepare` 的视觉容量。
- 人脸 crop 编码后，上传与关键点、前置判断、质量分级并行；两分支汇合，且质量合格、上传成功后才提取人脸特征，整个准备阶段仍占 Face 准备名额。人型先完成裁剪、质量和编码，再并行执行“上传后匹配人脸”与 SBS/TBPS 特征提取；上传仅使用原 `storage_io` 额度，不占人物视觉名额，两分支汇合后才按原检测顺序提交。0 级或空编码样本跳过人体特征分支，仍先完成上传步骤，再以 `enable_recognition=False` 分析脸部朝向。上传任务仍属于当前目标的 N+M 窗口，但同时等待上传的数量可能增加；未增加配置，原质量门槛、上传失败处理和身份提交顺序保留。
- `run_owned_parallel()` 持有每个目标的并行分支；未捕获异常或调用方取消时，取消其余分支并等待清理完成，重复取消不会截断清理。它只负责协程生命周期，已提交的 SDK 线程仍由原执行池持有到真实结束才归还许可，不保证停止已开始的网络调用。后台 writer/publisher 分别处理向量与 Kafka 队列，前台达到批量阈值只唤醒任务，不再等待整批写出。首次身份向量持久化后放行新生识别记录的规则保留。
- 人型身份检索在短锁内固定排除集合与版本后，可在锁外预取 Milvus 候选；重新加锁读取最新 pending 并复核状态，再按原检测顺序决定身份。候选不可用但当前可取得稳定版本时，最多释放锁刷新一次；仍不可用、实际写线程活跃或缺少版本能力时，保留完整锁内查询。决策到 pending 登记保持原子性，不等待 writer 空闲或无限重试。SBS 查询使用强一致读取，同一原型跨 Milvus/pending 按 UUID 去重；阈值、入库条件和新身份记录放行条件不变。并发与边界见[架构说明](代码架构与处理流程.md#64-特征准备与按序-reid-提交)。
- `ApplicationRuntime` 创建并关闭四个资源池；业务只接收所需的执行端口。图片 CPU、存储 I/O、Milvus 查询与写入分别限额，存储上传排队不会耗尽向量查询的线程许可。每池只把取得名额的工作提交给线程，不在 executor 内堆积无界任务；上游等待仍由消息、图片和目标窗口约束。
- 所有异步 Face 查询统一经过 `VectorQueryCoordinator`，在单进程或多进程 owner 内合并完全相容的请求，使用同一个 `vector_search` 池，不增加线程。默认批大小 4、凑批 1 ms、接纳容量 128；批大小为 1 也保留相同的接纳、取消和关闭路径。异步图库未注入调度器时明确报错，不回退到另一条搜索路径。client、维度及全部 SDK 查询参数一致才可合批，保持原 `Strong`、过滤和召回额度；SBS 身型分支继续独立查询。取消排队查询会撤销该项，已提交查询保持额度到 SDK 真实结束；关闭时先排空调度器，再关闭搜索池和数据库。容量之外的等待者仍受上游消息/目标窗口约束，这不是总 RSS 限制，也没有新增端到端查询超时。
- 数据库服务创建时先在当前进程预热 PyMilvus 的可选 SciPy 类型检查，避免首批并发查询同时初始化其惰性模块。该步骤不连接外部服务；SciPy 未安装时沿用 SDK 行为，已安装但初始化损坏时让启动失败。适配固定的 `pymilvus==3.0.1` 内部接口，升级 SDK 时须复核，详见[启动顺序](代码架构与处理流程.md#31-启动顺序)。
- `VISUAL_WORKER_PROCESSES=N`（1..4）时，通过 `spawn` 启动 N 个视觉进程，各自持有图像 CPU、存储、Triton client 和本地模型调度器；owner 仅持有 Milvus 搜索/写入池、身份状态、输出缓冲与全局准入。`N=1` 也使用 owner 与视觉进程分离的模式；`0` 使用有效的单进程装配，可供受限环境或受控对照使用。图像、mask 和裁图字节不经过身份 IPC；整图向量、人脸、人型仍按原顺序处理，Face 身份结果返回后才进入同帧 Person。各进程的 Triton 组批队列独立，实际 batch 与吞吐需回放验证。
- 多进程下这些配置继续表示全局资源预算：`IMAGE_CPU_WORKERS`、有效 `STORAGE_IO_WORKERS`、Triton client/执行/排队容量、Face/Person 全局准备容量、有效 `PERSON_FACE_CONCURRENCY` 按商余分配，子进程额度之和等于原配置；这些正数额度必须至少为 N，启动时拒绝无法分配的配置。每帧准备并发取原值与本进程全局容量的较小值，前瞻 M 保持原值；消息、图片数量和图片字节预算由 owner 统一控制，不复制到子进程。`TRITON_SYSTEM_SHM_SLOTS` 也按商余分配，单槽大小不变，因此总 SHM 字节数不变；总复制线程数保持 `min(4, 全局槽数)` 并分给各进程。某进程分到 0 槽时使用普通 gRPC。
- 消息按视觉进程当前在途消息数分配，没有按 CPU 配额加权。例如总 `IMAGE_CPU_WORKERS=4` 时，3 个视觉进程得到 `2/1/1`，4 个得到 `1/1/1/1`；相同在途消息数不代表相同处理能力。增加进程数还会分散本地组批，并增加进程、客户端和缓存开销；不要仅凭核心数决定进程数。`MULTIPROCESS_IPC_MAX_REQUESTS/MAX_BYTES` 保持每个 peer 的每类额度，随视觉进程数增加，整组 IPC 额度也增加，不能称所有资源总量都保持不变。
- 多进程视觉日志与 owner 分开：性能日志为 `PIPELINE_STATS_PATH` 同目录的 `visual-1.log` 至 `visual-N.log`；阶段日志在原文件名中加入 `.visual-1` 至 `.visual-N`。owner 的图片完成计数通过显式 IPC 汇总，子进程只返回自己的队列快照，避免重复消费周期性能计数。Snowflake ID 在当前单 owner 进程组内使用 owner worker ID 0、视觉 worker ID 1..N（最多 4）；这不是多个独立 VisionFlow 副本之间的 ID 分配协议，仍不支持直接复制多个相同编号的进程组写同一业务数据域。
- 服务启动调用 `cv2.setNumThreads(1)`，限制 OpenCV 算子内部线程与应用线程池竞争。资源池容量与 Triton RPC 容量独立；取消异步调用方不会提前归还仍在执行的线程许可，停机等待实际线程结束后关闭数据库连接。CPU/GPU 利用率与完整图片吞吐仍需用相同输入实测。
- 图片解码及人型准备要求显式注入 `image_executor`，统一使用 `IMAGE_CPU_WORKERS` 管理的图片资源域；独立朝向评估器也显式创建、注入和关闭 `AsyncWorkExecutor`。原 `IMAGE_DECODE_WORKERS`、`PERSON_PREPARATION_WORKERS`、`IO_THREAD_WORKERS` 及两套惰性图片池已移除，升级时删除旧配置，设置正值 `STORAGE_IO_WORKERS`。生产入口不设置 loop 默认执行器；重启后核对 `queues.resources` 的有效容量和等待。
- 慢裁图上传阻塞原图下载时，可设置 `STORAGE_DOWNLOAD_RESERVED_WORKERS`。下载仍可使用整个存储池；上传在事件循环等待分类名额，同一池线程总数保持不变，下载空闲时保底名额也不会借给上传。该值必须小于有效存储容量；多进程下按商余分配，启用时下载保底和剩余上传容量均至少为视觉进程数，避免某进程没有下载保底或无法上传。例：总存储64、四视觉、保底4时，每进程16线程，上传至多15，下载至少留1。默认4；调整时需比较完整图片吞吐、分类等待和输出排空；提高保底可能降低上传吞吐。取消调用不会提前归还已运行上传的名额，关闭会唤醒未接纳等待者并排空真实工作。
- `FrameBudget` 在下载之前保留图片名额，主图/全景图各自计数，图像流程完成或取消时释放。启用字节预算后，有效图片容量是配置帧数与 `FRAME_MEMORY_BUDGET_BYTES // FRAME_RESERVATION_BYTES` 中的较小值。它使用固定估算，不会在已持有图片时等待扩容；原图与压缩字节只作观测，不包括 mask、crop、Tensor、SDK 缓冲和仍运行的取消线程，因此不限制全进程 RSS。默认字节预算为 0，延续仅按数量接纳的行为。
- 人脸与人物的准备名额分别计数，推理 RPC 共同受 `TRITON_EXECUTOR_CONCURRENCY` 限制。两类 `FRAME_CONCURRENCY`、`GLOBAL_CONCURRENCY` 均须为正整数，两类 `PREPARE_LOOKAHEAD`、消息/图片容量覆盖值、`PERSON_FACE_CONCURRENCY` 和字节预算须为非负整数。调整准备并发与前瞻容量时，同时比较按序等待、消息在途、输出积压和完整图片吞吐，避免只把排队转移到下游。
- `TRITON_EXECUTOR_CONCURRENCY` 不得超过 `TRITON_CLIENT_POOL_SIZE`。
- `TRITON_SYSTEM_SHM_SLOTS>0` 要求 VisionFlow 与 Triton 位于同机且共享 `/dev/shm`，以及 Triton 使用 `--allow-client-shm=true`。所附 Compose 的 `ALOG_IPC_MODE` 默认 `private`；两端均使用 host IPC 时，将 VisionFlow 的该值设为 `host`，并重建容器。该方式只优化 `face_det`、`person_seg`、`person_pose` 大于等于 1 MiB 的 `RAW_IMAGE` 输入，输出仍走 gRPC。小图、超过单槽容量或没有可用槽时使用原传输路径。
  每个请求独占一个槽，图像复制在线程池执行。Triton 重启导致注册丢失时，只对当前槽的精确 `NOT_FOUND` 错误查询注册表；确认自身 region 缺失后，在保留原租约和像素的前提下重新注册同一 region，并最多重发一次。状态不明确、修复失败、其他已发送异常或取消仍将槽隔离直到 transport 关闭，后续重试成功也不恢复；总槽数固定，不补建，也不回收历史隔离槽。关闭会排空本地复制、请求和已提交的修复注册，只注销本进程创建的区域。开启后需对照完整图片吞吐，不能将单模型 RPC 速度直接视为业务 QPS。
  代码和模板保留 `TRITON_SYSTEM_SHM_SLOTS=0`、Compose `ALOG_IPC_MODE=private`，支持跨机器部署。本机同机配置使用 16 槽、每槽 32 MiB，共 512 MiB，并将两端 IPC 设为 `host`；多进程下槽数按全局预算分账，复制线程总计最多 4 个。该设置最多同时服务 16 个大图 RPC，其他调用即时使用普通传输，实际吞吐、事件循环延迟及内存仍需回放核对。
- `TRITON_BATCH_TARGETS`、`TRITON_BATCH_WAIT_MS` 使用 JSON 对象按[模型业务 key](代码架构与处理流程.md#81-triton-模型契约)覆盖配置，例如 `{"person_seg":4}`、`{"person_seg":20}`。
- 目标 batch 大于 `1` 时，对应等待时间必须大于 `0`，且目标不得超过 Triton 远程模型的 `max_batch_size`。默认目标为 B4，VisionFlow 启动校验远程模型声明的批上限；服务端还须保证 TensorRT engine 实际支持所用批大小。若服务端只支持 B1，须显式降低相应客户端目标，不能只修改服务端声明。
- 在曾用于回放的 Jetson Triton 配置中，Pose/Seg 使用两个 GPU 实例并启用 `dynamic_batching {}`；服务端可进一步合并来自不同视觉进程的请求。该服务端配置不包含在本仓库中，实际能力取决于自行部署的模型仓库。客户端 batch、服务端请求数和 TensorRT 实际 enqueue 数须分别观测，模块边界见[推理流程](代码架构与处理流程.md#82-单次推理请求的内部流程)。
- 降低服务端 `max_batch_size` 前，先降低客户端目标并重启 VisionFlow；提高服务端上限后，客户端目标保持原配置。
- 执行器队列满时立即拒绝新 batch。排队超时与远程推理超时分别计算；RPC 开始后使用远程推理期限。

### 业务选项

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `PIPELINE_STATS_ENABLED` | `true` | 是否写独立阶段统计文件；不影响业务背压 |
| `PIPELINE_STATS_PATH` | `logs/pipeline_stats.jsonl` | JSONL 日志路径，相对进程工作目录；100 MB 轮转、保留 7 天 |
| `PIPELINE_STATS_INTERVAL_SECONDS` | `5` | 阶段快照间隔，有限正数，秒；修改配置后重启 |
| `PERSON_FACE_REUSE_MODE` | `on` | `on` 复用同帧整图人脸的可靠身份、分析结果和特征，失败时本地回退；`off` 紧急关闭复用并完整处理 crop 内人脸 |
| `PERSON_UPRIGHT_DEVICE_SNS` | 空 | 额外启用人物转正的设备 SN，逗号分隔；留空表示没有额外设备 |

`panoKeyframeUrl` 来源和 SN 含 `EPS`（忽略大小写）的设备自动启用人物转正，不受手动名单清空影响。转正仅在可靠 Pose 条件下执行；成功旋转的人物使用最终 crop 内的本地人脸流程。同图复用的可靠性和回退条件见[人脸复用规则](人脸人型详细规则.md#同帧整图人脸复用状态机)。

输出 Topic 可通过 `KAFKA_FACE_DETECTION_RECORD_TOPIC`、`KAFKA_FACE_RECOGNITION_RECORD_TOPIC`、`KAFKA_PERSON_DETECTION_RECORD_TOPIC`、`KAFKA_PERSON_RECOGNITION_RECORD_TOPIC`、`KAFKA_VECTOR_RECORD_TOPIC` 覆盖；默认名称和字段见 [Kafka Topic 契约](kafka_topic.md)。

### 隔离回放命名空间

`MILVUS_COLLECTION_PREFIX` 默认为空；非空时直接加到四个集合名之前，例如 `replay_` 对应 `replay_alogFace`。它创建/检查所选命名空间，不迁移、回退或删除原集合，仍保留不兼容结构检查。`STORAGE_OUTPUT_PREFIX` 默认为空；非空时只改变新裁图 object key，原图仍按输入 key 从同一所选存储后端读取，输出和 Milvus 元数据保存实际带前缀的裁图 key。

回放时一起设置测试 `KAFKA_INPUT_TOPICS`、独立 `KAFKA_GROUP_ID`、五类输出 Topic、Milvus 集合前缀和裁图前缀。只有换输入 Topic 不能隔离身份写入。A/B 各轮应恢复同一测试图库初始状态；这些开关不会自动制作快照、重置数据或保证跨服务事务。在隔离环境中自行准备回放工具并确认输出与写入均已排空。

## 输入与处理结果

服务默认订阅 `PanoramicPersonDetSync`、`PtzPersonDetSync`、`PtzFaceDetSync`；`KAFKA_INPUT_TOPICS` 可用非空、无重复的逗号分隔列表覆盖。各输入 Topic 使用相同流程。消息 value 为 UTF-8 JSON，例如：

```json
{
  "sn": "demo-camera-001",
  "timestamp": 1710000000000,
  "uId": "11111111-1111-4111-8111-111111111111",
  "keyframe": "demo-camera-001/picture/frame.jpg"
}
```

`uId` 必须是非空字符串，输出时映射为 `keyframeId`。可选的 `panoKeyframeUrl` 在主图处理成功后作为第二张图片处理；两图共享业务 ID，各自生成 `pictureVectorId`，并由 `isPanoKeyframe` 区分来源。

当前不执行设备 SN 白名单过滤，消息通过字段和对象存储地址解析后即进入处理。设备 SN 仍用于输出上下文，以及 `EPS` 和 `PERSON_UPRIGHT_DEVICE_SNS` 的人物转正判断。字段别名和对象存储地址解析规则见 [Kafka 输入契约](kafka_topic.md#0-输入-topic)。

每张图片按“整图向量 → 人脸 → 人型”处理。人脸与人型默认跨设备检索，质量等级决定能否识别和入库；SBS 负责 ReID 分配，TBPS/CLIP 用于检索。SBS 新建或扩充还要求当前观察到非零人脸 ID。

| Collection | 内容 |
| --- | --- |
| `alogPictureClip` | 512 维整图向量 |
| `alogFace` | 512 维人脸向量及质量信息 |
| `alogSBS` | 2048 维人物身份原型 |
| `alogTBPS` | 512 维人物检索向量 |

Triton 必须提供[模型契约表](代码架构与处理流程.md#81-triton-模型契约)中的 9 个模型。客户端通过 gRPC 发现模型就绪状态、契约版本、Tensor 结构和 batch 上限；不兼容时初始化失败。`alogSBS` 的字段和主键要求见 [SBS 集合结构](人型ReID识别完整流程.md#13-alogsbs-schema)，结构不兼容时服务不会自动删除集合。

输入 offset 在消息入本地队列后提交，不等待业务完成。输出由缓冲区批量写出；新生 Face ID/ReID 的识别记录等待首次向量持久化确认，已有身份命中记录直接入队。正常停机先等待本地队列 `join()` 确认真正处理完成，再关闭消息调度器，最后停止缓冲 writer/publisher。输出由单进程运行时或多进程 owner 持有，没有持久 outbox 或新的 offset 协议；进程异常退出可能丢失尚未刷出的数据，各输出 Topic 没有事务性顺序保证。

## 链路计时与 Kafka 回放

运行时每 5 秒输出一条 `ALog performance summary: {JSON}`，汇总原图解码 `image_decode`、解码排队 `image_decode_queue_wait`、人型分割响应解包及 mask 重建 `person_seg_unpack` 和事件循环调度延迟 `event_loop_lag` 的次数、累计耗时、平均耗时和最大耗时。`image_decode` 只测量工作线程中的转换与解码，排除下载、名额等待和线程调度；`image_decode_queue_wait` 从下载完成后调用解码起，到工作线程开始解码为止。取消后仍运行的解码也会记录，所有聚合计数都在事件循环线程更新。分割解包仍同步执行；事件循环延迟使用周期探针测量，不能直接当作某个业务函数耗时。

人物裁图准备由 `person_crop_prepare` 和 `person_crop_prepare_queue_wait` 分别统计实际计算与排队耗时。四个资源域同时记录 `<name>_run/queue_wait`，与业务分项存在重叠，不能相加。身份锁以 `face_identity_lock_wait` / `person_identity_lock_wait` 统计成功获取前的等待，对应 `*_hold` 统计锁内完整处理。Face 计时由身份服务统一记录，覆盖独立人脸和人物 Face 复用/回退；相比旧版本新增了后者，A/B 对比需注明口径变化。

`person_observation_build` / `person_observation_build_queue_wait` 分别统计每帧分割/Pose 绑定、朝向估计与初始结构质量构建的实际执行和排队。正式入口每帧向 `image_cpu` 提交一次，保持原检测顺序；不含推理请求及输出解包。日志和指标聚合仍在事件循环执行，独立朝向评估器也显式注入并持有图片执行器。这两项同样是 `image_cpu_*` 的分项，不能相加。

人型的 `person_identity_requests` 统计有效 embedding 的身份请求开始次数；`person_identity_locked_search` 直接统计最终完整锁内查询的次数和耗时。`person_identity_lock_hold` 不含锁外查询，包含最终回退；前置短锁单独计为 `person_identity_snapshot_lock_wait/hold`。`person_identity_prefetch` 包含初次预取和刷新，`person_identity_refresh` 单列额外刷新，不能相加。`person_identity_refresh_used/invalid/failed/unavailable/exhausted` 分别计刷新通过、失效、异常、当前不能刷新及刷新后仍回退。初次预取普通异常记录 `person_identity_prefetch_failed` 后传播；刷新普通异常记录 failed 后进入原完整回退；取消直接传播。

`person_identity_prefetch_used/fallback/skipped` 只读取 count，fallback 现在表示候选复核失效，随后可能刷新，不等于锁内查询。一次逻辑请求最多经历两次提交锁等待/持有；计算总锁占用须合计快照与提交 hold，不能用旧版 used/fallback/skipped 合计推算请求数。`person_identity_search` 计每次持锁搜索段，按逻辑请求计算平均值时使用 requests 作分母。

Face 的 `face_identity_requests` 记录逻辑解析开始次数，包含独立人脸提交和人物内部只读查询，以及后来失败/取消的请求；`face_identity_locked_search` 直接统计最终完整锁内查询次数和耗时。初次短锁计为 `face_identity_snapshot_lock_wait/hold`，`face_identity_prefetch` 包含初次和额外刷新两种锁外查询，`face_identity_refresh` 单列额外刷新；两者耗时有重叠，不能相加。刷新结果用 `face_identity_refresh_used/invalid/failed/unavailable/exhausted` 的 count 区分。

`face_identity_prefetch_used/fallback/skipped/failed` 计候选复核或预取尝试；fallback 表示候选失效，随后可能锁外刷新，并不直接等于完整锁内查询。一次逻辑请求可有两次提交锁等待/持有和两次候选失效，不能用旧版 used/fallback/skipped 之和推算请求或锁内查询数。关闭预取时仍记录 requests、locked_search 和 refresh_unavailable；预取异常本身不能当作正常未命中，取消直接传播。`face_milvus_search`、`face_pending_search`、`face_match_resolve` 分别统计远程查询、pending 查询和候选解析，包含适用的初次预取、刷新和回退调用，不能将并行耗时相加当作吞吐或服务端执行时间。

人脸身份锁内的完整搜索另按 `face_identity_locked_full_pending/milvus/resolve` 分段，预取候选复核按 `face_identity_locked_prefetch_pending/resolve` 分段；这些计时只覆盖身份服务持锁协程中使用 `FaceSearchService` 的调用，包含异步排队和调度等待，不是 CPU 时间或 Milvus 服务端执行时间。分项嵌套在身份锁持有和原有查询指标内，不能与它们相加；自定义搜索回调及跨周期样本可能使分项计数与完整搜索数不同。。

锁内人脸查询的 Milvus 客户端阶段进一步记录 `face_identity_locked_query_admission/batch/executor/sdk/resume`，分别覆盖接纳、入队到选批、选批到工作线程启动、SDK 调用、SDK 返回到调用方恢复；这些按锁内向量请求计数，同一 RPC 批次的耗时可能在多个请求中重复。它们仍是客户端墙钟时间，不能直接与服务端指标逐请求配对，也不能将累计值当作独立 RPC 的总耗时。

SBS 并行检索使用显式注入的 `vector_search` 资源后统一等待，保持每次查询的身型分支并行、结果顺序和上下文隔离。`sbs_search_group_wait` 统计整组等待；成功组按最后执行 Future 完成回调的分支，拆为 `future_critical_dispatch`、`future_critical_thread_queue`、`future_critical_thread_run`、`future_critical_callback` 和 `future_group_join`（均带 `sbs_search_` 前缀）。五段合计本组等待，回调时间不等同于逐分支 `await` 恢复。关联状态仅用于本次检索组，不按请求或身份长期保存，仍通过每 5 秒的性能汇总输出，不记录身份或逐请求日志。。

Face 查询统一调度器的 `queues.resources.vector_queries` 提供已接纳/排队向量数、在途批次、实际完成 RPC 数和 batch 直方图。`vector_query_admission_wait` 与 `vector_query_batch_wait` 按单向量计时；`vector_query_rpc_run` 按真实 RPC 计时，与 `vector_search_run` 有重叠。不能仅凭旧搜索池排队下降判断收益，应同时检查调度器前段等待、每图 RPC 数和完整吞吐；不同计数单位的平均值不能直接相加。

开启 system SHM 后，`triton_shm_copy_queue` / `triton_shm_copy_run` 分别统计复制前等待和线程内复制耗时。`triton_shm_used`、`triton_shm_fallback`、`triton_shm_poisoned` 只读取 `count`；其零值耗时没有延迟意义。`fallback` 包含小输入、其他模型与槽不可用等全部原传输调用，以及隔离槽后的原传输重试，不能解释为槽不足率。`triton_shm_reregister_attempt/reregistered/reregister_failed` 分别统计注册修复流程进入、完成并允许重发、拒绝或失败（含取消）的次数，也只读取 `count`。修复后的 SHM 重发另计一次 `used` 和 `rpc_attempts`；控制查询与注册计入 `rpc_ms`，不计为推理 RPC 尝试。逐请求 DEBUG 明细的 `serialize_ms` 在 SHM 路径还包含复制排队、复制和事件循环恢复的墙钟时间，不能当成事件循环被阻塞的时间。

推理组包仅在预计复制至少 1 MiB 的非连续数值数组或多样本数值堆叠时进入共享 `image_cpu`，连续单图和 encoded bytes 包装保留轻量同步路径。逐请求 DEBUG 的 `pack_ms` 统计实际组包，`pack_queue_wait_ms` 单列 CPU 等待并计入 `queue_total_ms`；这些字段尚未加入 `inference_by_model` 周期汇总，CPU 域执行仍有资源聚合计时。它允许大数组复制期间调度其他工作，不保证单次推理更快；输出解包保持原执行位置。

批次等待、RPC 等推理耗时按模型汇总。业务消息处理完成、图片流程完成和 Milvus 向量写入分别统计；一条消息可包含主图和全景图，一张图片可生成多条向量。性能汇总不代表所有异步写库及 Kafka 输出已经完成。

本仓库不包含 Kafka 快照与回放工具；进行性能比较时需自行准备隔离输入及图库。

