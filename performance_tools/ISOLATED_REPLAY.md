# 隔离历史 Kafka 回放

`isolated_replay.py` 为当前重构版本装配独立测试命名空间，复用 `kafka_replay.py` 的有限、限速回放和完成追踪。`plan` 完全离线；`snapshot-gallery` 只读 Milvus；`prepare` 创建测试资源；`worker` 与 `replay` 会触发真实业务处理。连接配置沿用 `core.config.settings`，运行时环境变量优先于项目根目录 `.env`，命令和产物清单不需要包含连接密钥。

每个命名空间包含一个输入 Topic、五个输出 Topic、独立 consumer group、四个 Milvus 集合和裁图前缀。集合固定以 `alogbench_` 开头，裁图固定以 `alogbench/` 开头。原图下载保留历史 object key，既不复制图片，也不回退其他存储后端。清理和复用旧命名空间不属于此脚本功能。

## 固定输入与初始图库

以下示例在项目根目录使用同一个 Python 环境执行。命名空间必须是 1..48 位小写字母、数字或下划线，首位为字母；每轮选择新名字。快照是本地产物，可能不随代码仓库分发。

```bash
.venv/bin/python performance_tools/isolated_replay.py plan \
  --namespace bench_a --output-dir exports/isolated-ab \
  --input performance_tools/results/qps6-snapshot/snapshot.jsonl

.venv/bin/python performance_tools/isolated_replay.py plan \
  --namespace bench_b --output-dir exports/isolated-ab \
  --input performance_tools/results/qps6-snapshot/snapshot.jsonl
```

`manifest.json` 保存来源路径、输入 SHA256、消息数量和派生资源名；不保存输入正文。后续命令验证命名规则，回放还会重新校验输入 SHA256。`plan` 不验证历史图片当前是否可下载。

目录改名前生成的旧 manifest 可能仍保存旧目录 `test_time/results/` 下的输入路径；更名不会自动改写这些历史产物。需要继续回放时，先核对原快照，再使用新路径重新生成 manifest，并按新 manifest 的命名空间准备测试资源。

暖图库比较只导出一次 Face/SBS，A/B 都恢复同一份文件。默认读取上限为 Face 20 行、SBS 1 行，查询 `limit + 1` 行；超过上限会失败，核对集合后显式提高上限，不能静默截取部分身份。冷库比较省略 `--warm-snapshot`，不要把冷库和暖库的结果混在一起。

```bash
.venv/bin/python performance_tools/isolated_replay.py snapshot-gallery \
  --face-source alogFace --sbs-source alogSBS \
  --face-limit 20 --sbs-limit 1 \
  --output exports/isolated-ab/warm-gallery.json

.venv/bin/python performance_tools/isolated_replay.py prepare \
  --manifest exports/isolated-ab/bench_a/manifest.json \
  --warm-snapshot exports/isolated-ab/warm-gallery.json

.venv/bin/python performance_tools/isolated_replay.py prepare \
  --manifest exports/isolated-ab/bench_b/manifest.json \
  --warm-snapshot exports/isolated-ab/warm-gallery.json
```

快照包含向量、身份和图片元数据，文件权限为 `0600`，不应提交。采集只使用 `describe_collection` 与有限的 Strong 查询，不修改、删除或重载生产集合；两次查询不构成跨集合事务快照。恢复保留主键、身份、向量和事件时间，并验证目标空库、插入回执和主键集合。`prepared.json` 记录 manifest 与暖图库文件哈希、恢复行数、Topic 分区及副本数。

`prepare` 默认一分区、一副本，可通过 `--partitions` 和 `--replication-factor` 调整。任何同名 Topic 或集合已经存在都会拒绝继续。部分失败可能留下本轮资源，换新命名空间再执行；脚本不会自动删除或重置旧数据、生产 offset。

## 顺序启动和回放

每次只运行一个 A/B Worker，以免两轮互相争用 CPU/GPU、Triton、Milvus 和对象存储。当前版本的启动命令如下；对比旧版本时需使用该版本经过核对的隔离装配入口，使其消费相同 manifest 指定的 Topic 并使用相同的初始图库。

```bash
.venv/bin/python performance_tools/isolated_replay.py worker \
  --manifest exports/isolated-ab/bench_a/manifest.json
```

Worker 前台运行，另一个终端发送有限样本：

```bash
.venv/bin/python performance_tools/isolated_replay.py replay \
  --manifest exports/isolated-ab/bench_a/manifest.json \
  --count 10 --qps 1 --wait-seconds 60
```

完成小规模连通性验证后，正式 A/B 使用新的命名空间，避免冒烟测试已经扩充图库。A/B 固定同一消息顺序、事件时间、模型、算法 TOML 和采样窗口，除明确列出的被测参数外保持有效配置一致。每个命名空间仅启动一个 Worker，脚本保留原事件时间、去掉 pano，并为每次发送创建可追踪的独立 `uId`。

每轮保存实际运行的代码版本或源码摘要、配置覆盖值和有效资源容量。隔离入口只设置测试命名空间、日志和消费策略，不固定 `IMAGE_CPU_WORKERS`、`STORAGE_IO_WORKERS`、消息/图片在途容量及组批参数；这些值仍可能来自环境变量、`.env` 或代码默认值。只保存命令中的覆盖值不能代表全部有效配置，也不能用后续修改后的默认值回填历史运行。

若正式轮之前需要预热，应让两侧执行相同输入、顺序和节奏，并将预热轮与计量轮分别记录。预热也会学习和扩充身份图库；`prepared.json` 的相同哈希只证明装载的初始快照相同，不能证明预热之后、正式计量开始时的图库严格相同。报告应注明动态图库与自然身份漂移，并结合逐目标结果判断；需要严格固定图库时，应另行设计只读身份模式或在正式轮之前恢复同一状态，当前脚本不会自动提供这些行为。

下面的输入压力示例每轮发 600 条；150 条快照会循环四遍。发送目标 50 条/秒不代表完整处理达到 50 张/秒，应结合实际完成量、尾部排空、积压和身份结果判断。

```bash
.venv/bin/python performance_tools/isolated_replay.py replay \
  --manifest exports/isolated-ab/bench_a/manifest.json \
  --count 600 --qps 50 --wait-seconds 150 \
  --max-inflight 20
```

`--max-inflight` 只限制 producer 尚未收到 Kafka ACK 的发送任务，与 Worker 的 `ALOG_MAX_INFLIGHT_MESSAGES`、`ALOG_MAX_INFLIGHT_FRAMES` 是不同容量。

可重复添加 `--metrics NAME=URL` 保存外部服务指标首尾快照。Worker 正常受控停机时等待业务任务及异步写出排空；异常退出或强制终止不具备同样保证。在确认完成并停止 A 后，再按相同过程运行 B。有限回放不重置生产消费组，也不会将生产 Topic 作为输入默认值。

## 结果与验收边界

每个 manifest 所在目录保存 `logs/ALog.log`、`pipeline_stats.jsonl` 和 `runs/<run_id>/`。每个 run 的 `sent.jsonl`、`outputs.jsonl`、`summary.json` 格式见 [回放工具说明](README.md)。普通 `outputs.jsonl` 只保留输出定位及延迟，不包含用于身份对照的完整身份和框；需要细粒度对照时，另行只读采集隔离输出 Topic。

完整处理和输出持久化是不同的完成边界。检查完成追踪的每张图片成功/失败计数，再检查 Kafka ACK、向量/记录队列排空和最后输出时间；观察时间结束时没有见到输出不能直接认定业务丢失。跨版本阶段统计还要核对指标口径。

测试命名空间隔离的是输入、输出与状态，底层硬件和服务仍可能共享。报告需记录同期其他流量。小规模循环样本只能说明该有限负载下的吞吐、等待和资源占用；没有多身份真值标注时，不能据此证明身份准确率。
