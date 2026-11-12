from pathlib import Path
from collections.abc import Iterable
from math import isfinite
import re
from typing import Annotated, Any, Literal
from urllib.parse import quote_plus, urlsplit

from pydantic import (
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
TRITON_MODEL_KEYS = frozenset({
    "clip_image",
    "face_rec",
    "tbps_image",
    "sbs_image",
    "face_qa",
    "face_kps",
    "face_det",
    "person_seg",
    "person_pose",
})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=False,
        extra="ignore",
        populate_by_name=True,
        validate_default=True,
    )

    LOG_LEVEL: str = "INFO"
    PIPELINE_STATS_ENABLED: bool = True
    PIPELINE_STATS_PATH: str = "logs/pipeline_stats.jsonl"
    PIPELINE_STATS_INTERVAL_SECONDS: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    # 在领取新消息前执行输出水位背压；已在途帧可继续提交，非队列硬上限。
    OUTPUT_BACKLOG_HIGH_WATERMARK: int = Field(default=2000, ge=1)
    OUTPUT_BACKLOG_LOW_WATERMARK: int = Field(default=1000, ge=0)
    # 额外手动启用人物转正的设备；SN 包含 EPS（忽略大小写）和全景图自动启用。
    person_upright_device_sns: Annotated[frozenset[str], NoDecode] = Field(
        default=frozenset(),
        validation_alias="PERSON_UPRIGHT_DEVICE_SNS",
    )
    person_face_reuse_mode: Literal["off", "on"] = Field(
        default="on",
        validation_alias="PERSON_FACE_REUSE_MODE",
    )

    PGSQL_USER: str
    PGSQL_PSWD: str
    PGSQL_HOST: str
    PGSQL_PORT: int
    PGSQL_DB: str

    KAFKA_BROKER_ADDR: str
    KAFKA_BROKER_PORT: int
    KAFKA_USERNAME: str | None = None
    KAFKA_PASSWORD: str | None = None
    KAFKA_GROUP_ID: str
    KAFKA_INPUT_TOPICS: Annotated[tuple[str, ...], NoDecode] = (
        "PanoramicPersonDetSync", "PtzPersonDetSync", "PtzFaceDetSync",
    )
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"
    KAFKA_SECURITY_PROTOCOL: str = "SASL_PLAINTEXT"
    KAFKA_SASL_MECHANISM: str = "PLAIN"
    KAFKA_FACE_DETECTION_RECORD_TOPIC: str = "visionflow-face-detection-record"
    KAFKA_FACE_RECOGNITION_RECORD_TOPIC: str = "visionflow-face-recognition-record"
    KAFKA_PERSON_DETECTION_RECORD_TOPIC: str = "visionflow-person-detection-record"
    KAFKA_PERSON_RECOGNITION_RECORD_TOPIC: str = "visionflow-person-recognition-record"
    KAFKA_VECTOR_RECORD_TOPIC: str = "visionflow-vector-record"

    TRITON_SERVER_URL: str
    # ALog 业务消息处理 worker 数，控制本地流水线并发，不等同于 Triton RPC 并发。
    ALOG_WORKER_COUNT: int = 32
    # 默认 owner + 4 个视觉进程；0 用于受限环境的单进程运行。
    VISUAL_WORKER_PROCESSES: int = Field(default=4, ge=0)
    MULTIPROCESS_IPC_MAX_REQUESTS: int = Field(default=128, ge=1)
    MULTIPROCESS_IPC_MAX_BYTES: int = Field(default=67108864, ge=1)
    MULTIPROCESS_IPC_MAX_PAYLOAD_BYTES: int = Field(default=4194304, ge=1)
    MULTIPROCESS_START_TIMEOUT_SECONDS: float = Field(default=90.0, gt=0, allow_inf_nan=False)
    MULTIPROCESS_STOP_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    # 0 沿用 worker 数；独立限制已接纳但业务尚未结束的消息，不包含 Kafka 本地队列。
    ALOG_MAX_INFLIGHT_MESSAGES: int = Field(default=100, ge=0)
    ALOG_MAX_INFLIGHT_FRAMES: int = Field(default=100, ge=0)
    # 固定估算预留用于下载前接纳；0禁用字节预算，不是进程RSS硬上限。
    FRAME_MEMORY_BUDGET_BYTES: int = Field(default=0, ge=0)
    FRAME_RESERVATION_BYTES: int = Field(default=67108864, ge=1)
    IMAGE_CPU_WORKERS: int = Field(default=4, ge=1)
    # 显式存储容量；不继承默认线程池或已删除的旧配置。
    STORAGE_IO_WORKERS: int = Field(default=64, ge=1)
    # 同一存储池内限制上传并发，为原图下载保留容量；0 保持原共享准入。
    STORAGE_DOWNLOAD_RESERVED_WORKERS: int = Field(default=4, ge=0)
    MILVUS_SEARCH_WORKERS: int = Field(default=16, ge=1)
    MILVUS_WRITE_WORKERS: int = Field(default=4, ge=1)
    # 所有异步 Face 查询统一经过有界调度器；1 仅表示每批一个查询。
    MILVUS_QUERY_BATCH_SIZE: int = Field(default=4, ge=1, le=32)
    MILVUS_QUERY_WAIT_MS: float = Field(default=1.0, ge=0, le=10, allow_inf_nan=False)
    # 同时约束已接纳排队与真实在途向量数量；满时背压到现有上游窗口。
    MILVUS_QUERY_CAPACITY: int = Field(default=128, ge=1)
    # 同机系统共享内存输入槽；0 关闭。显式启用需共享 IPC 与服务端许可。
    TRITON_SYSTEM_SHM_SLOTS: int = Field(default=0, ge=0, le=64)
    TRITON_SYSTEM_SHM_SLOT_BYTES: int = Field(default=33554432, ge=1048576)
    # 每帧同时准备的人脸上限 N；未提交总容量为 N + FACE_PREPARE_LOOKAHEAD。
    FACE_FRAME_CONCURRENCY: int = Field(default=30, ge=1)
    # 同一个人脸处理器的全部消息 worker 共享的准备与推理并发上限。
    FACE_GLOBAL_CONCURRENCY: int = Field(default=100, ge=1)
    FACE_PREPARE_LOOKAHEAD: int = Field(default=2, ge=0)
    FACE_SEARCH_PREFETCH_ENABLED: bool = True
    # 每帧同时准备的人物上限 N；未提交总容量为 N + PERSON_PREPARE_LOOKAHEAD。
    PERSON_FRAME_CONCURRENCY: int = Field(default=30, ge=1)
    # 全部消息共享的人物视觉准备与 embedding 并发上限；Face解析另计容量。
    PERSON_GLOBAL_CONCURRENCY: int = Field(default=100, ge=1)
    # 0沿用人物全局并发；Face解析不占人物视觉准备名额。
    PERSON_FACE_CONCURRENCY: int = Field(default=0, ge=0)
    PERSON_PREPARE_LOOKAHEAD: int = Field(default=2, ge=0)
    # 默认真实组批；目标不得超过远程模型声明的 max_batch_size。
    TRITON_BATCH_TARGET_DEFAULT: int = 4
    # 未按模型覆盖时的最大凑批等待时间，单位毫秒；0 表示不额外等待。
    TRITON_BATCH_WAIT_MS_DEFAULT: float = 2.0
    # 按稳定模型 key 覆盖目标批大小；未配置的模型使用 TRITON_BATCH_TARGET_DEFAULT。
    TRITON_BATCH_TARGETS: dict[str, int] = Field(default_factory=dict)
    # 按稳定模型 key 覆盖凑批等待毫秒数；目标批大于 1 时对应等待必须大于 0。
    TRITON_BATCH_WAIT_MS: dict[str, float] = Field(default_factory=dict)
    # 从 scheduler submit 起计算的绝对排队期限（秒），覆盖凑批和 executor 等待。
    # batch 一旦进入执行 lane，此期限不再中断已经开始的远程调用。
    TRITON_QUEUE_TIMEOUT_SECONDS: float = 60.0
    # 从共享客户端池获取一个 Triton gRPC 客户端的最长等待时间，单位秒。
    TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS: float = 5.0
    # 单次远程 inference RPC 开始后的超时时间，单位秒。
    TRITON_INFER_TIMEOUT_SECONDS: float = 30.0
    # 启动阶段获取 Triton 模型配置和元数据的单次 RPC 超时时间，单位秒。
    TRITON_CONFIG_TIMEOUT_SECONDS: float = 5.0
    # 可复用 Triton gRPC 客户端数量；不得小于 TRITON_EXECUTOR_CONCURRENCY。
    TRITON_CLIENT_POOL_SIZE: int = 100
    # 全部模型共享的远程 batch 执行 lane 数，即同时进行的推理 RPC 上限。
    TRITON_EXECUTOR_CONCURRENCY: int = 80
    # executor 内等待 lane 的已形成 batch 上限，不包含正在执行的 batch；满时立即拒绝。
    TRITON_EXECUTOR_QUEUE_CAPACITY: int = 256
    # InferenceUnavailable 类瞬时错误的额外重试次数；当前只允许 0 或 1。
    TRITON_TRANSIENT_RETRIES: int = 0

    MILVUS_URI: str
    MILVUS_TOKEN: str
    MILVUS_COLLECTION_PREFIX: str = ""
    STORAGE_OUTPUT_PREFIX: str = ""

    STORAGE_BACKEND: Literal["oss", "minio"] = "oss"
    OSS_ACCESS_KEY_ID: str | None = None
    OSS_ACCESS_KEY_SECRET: str | None = None
    OSS_ENDPOINT: str | None = None
    OSS_BUCKET_NAME: str | None = None
    MINIO_ENDPOINT: str | None = None
    MINIO_ACCESS_KEY: str | None = None
    MINIO_SECRET_KEY: str | None = None
    MINIO_BUCKET_NAME: str | None = None
    MINIO_SECURE: bool = False

    @field_validator("*", mode="before")
    @classmethod
    def _reject_blank_strings(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                if info.field_name in {
                    "person_upright_device_sns", "MILVUS_COLLECTION_PREFIX", "STORAGE_OUTPUT_PREFIX",
                }:
                    return value
                if info.field_name in {
                    "KAFKA_USERNAME", "KAFKA_PASSWORD",
                    "OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET",
                    "OSS_ENDPOINT", "OSS_BUCKET_NAME",
                    "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY",
                    "MINIO_BUCKET_NAME",
                }:
                    return None
                raise ValueError("must not be empty")
        return value

    @field_validator("MILVUS_QUERY_BATCH_SIZE", "MILVUS_QUERY_WAIT_MS", "MILVUS_QUERY_CAPACITY", mode="before")
    @classmethod
    def _reject_boolean_query_limits(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("query limits cannot be boolean")
        return value

    @model_validator(mode="after")
    def _validate_query_limits(self):
        if self.MILVUS_QUERY_CAPACITY < self.MILVUS_QUERY_BATCH_SIZE:
            raise ValueError("MILVUS_QUERY_CAPACITY must be >= MILVUS_QUERY_BATCH_SIZE")
        return self

    @field_validator("KAFKA_INPUT_TOPICS", mode="before")
    @classmethod
    def _input_topics(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            value = value.split(",")
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("must be a nonempty CSV or list of topic names")
        topics = tuple(item.strip() if isinstance(item, str) else "" for item in value)
        if any(not re.fullmatch(r"[A-Za-z0-9._-]{1,249}", item) or item in {".", ".."} for item in topics):
            raise ValueError("invalid Kafka topic name")
        if len(set(topics)) != len(topics):
            raise ValueError("duplicate Kafka input topics")
        return topics

    @field_validator("MILVUS_COLLECTION_PREFIX")
    @classmethod
    def _collection_prefix(cls, value: str) -> str:
        if value and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value):
            raise ValueError("must be empty or a collection-name prefix of at most 128 characters")
        return value

    @field_validator("STORAGE_OUTPUT_PREFIX")
    @classmethod
    def _output_prefix(cls, value: str) -> str:
        value = value.strip("/")
        if value and (len(value) > 128 or any(part in {"", ".", ".."} for part in value.split("/"))
                      or not re.fullmatch(r"[A-Za-z0-9_/-]+", value)):
            raise ValueError("must be a relative object-key prefix of at most 128 characters")
        return value

    @model_validator(mode="after")
    def _validate_storage_reserve(self) -> "Settings":
        if self.STORAGE_DOWNLOAD_RESERVED_WORKERS >= self.STORAGE_IO_WORKERS:
            raise ValueError("STORAGE_DOWNLOAD_RESERVED_WORKERS must be smaller than STORAGE_IO_WORKERS")
        return self

    @field_validator("STORAGE_DOWNLOAD_RESERVED_WORKERS", mode="before")
    @classmethod
    def _reject_boolean_storage_reserve(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("storage download reserve must not be boolean")
        return value

    @model_validator(mode="after")
    def _validate_frame_budget(self) -> "Settings":
        if 0 < self.FRAME_MEMORY_BUDGET_BYTES < self.FRAME_RESERVATION_BYTES:
            raise ValueError("FRAME_MEMORY_BUDGET_BYTES must cover at least one FRAME_RESERVATION_BYTES")
        return self

    @model_validator(mode="after")
    def _validate_visual_workers(self) -> "Settings":
        if not 0 <= self.VISUAL_WORKER_PROCESSES <= 4:
            raise ValueError("VISUAL_WORKER_PROCESSES must be between 0 and 4")
        if self.MULTIPROCESS_IPC_MAX_PAYLOAD_BYTES > self.MULTIPROCESS_IPC_MAX_BYTES:
            raise ValueError("MULTIPROCESS_IPC_MAX_PAYLOAD_BYTES must not exceed MULTIPROCESS_IPC_MAX_BYTES")
        if self.VISUAL_WORKER_PROCESSES:
            from core.multiprocess.assembly import validate_visual_budgets

            validate_visual_budgets(self, self.VISUAL_WORKER_PROCESSES)
        return self

    @field_validator(
        "VISUAL_WORKER_PROCESSES", "MULTIPROCESS_IPC_MAX_REQUESTS",
        "MULTIPROCESS_IPC_MAX_BYTES", "MULTIPROCESS_IPC_MAX_PAYLOAD_BYTES",
        "MULTIPROCESS_START_TIMEOUT_SECONDS", "MULTIPROCESS_STOP_TIMEOUT_SECONDS",
        mode="before",
    )
    @classmethod
    def _reject_boolean_multiprocess_limits(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("multiprocess capacities and timeouts must not be boolean")
        return value

    @model_validator(mode="after")
    def _validate_storage_settings(self) -> "Settings":
        required = (
            ("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET_NAME")
            if self.STORAGE_BACKEND == "oss" else
            ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_NAME")
        )
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(
                f"{self.STORAGE_BACKEND} storage requires: {', '.join(missing)}"
            )
        if self.STORAGE_BACKEND == "minio":
            endpoint = self.MINIO_ENDPOINT
            try:
                parsed = urlsplit(f"//{endpoint}")
                port = parsed.port
                valid = (
                    bool(parsed.hostname) and not parsed.username and not parsed.password
                    and not parsed.path and not parsed.query and not parsed.fragment
                    and not any(char.isspace() for char in endpoint)
                    and (port is None or 1 <= port <= 65535)
                    and not endpoint.endswith(":")
                )
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("MINIO_ENDPOINT must be host[:port] without scheme, path or credentials")
        return self

    @model_validator(mode="after")
    def _validate_output_watermarks(self) -> "Settings":
        if self.OUTPUT_BACKLOG_LOW_WATERMARK >= self.OUTPUT_BACKLOG_HIGH_WATERMARK:
            raise ValueError("OUTPUT_BACKLOG_LOW_WATERMARK must be lower than OUTPUT_BACKLOG_HIGH_WATERMARK")
        return self

    @field_validator("MINIO_SECURE", mode="before")
    @classmethod
    def _ignore_inactive_minio_tls(cls, value: Any, info: ValidationInfo) -> Any:
        if info.data.get("STORAGE_BACKEND") == "oss":
            return False
        return value

    @field_validator("person_upright_device_sns", mode="before")
    @classmethod
    def _normalize_device_sns(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            values = value.split(",")
        elif isinstance(value, Iterable):
            values = tuple(value)
            if not all(isinstance(raw_item, str) for raw_item in values):
                raise ValueError("must contain only strings")
        else:
            raise ValueError("must be a CSV string or an iterable of strings")
        return frozenset(
            item
            for item in (raw_item.strip() for raw_item in values)
            if item
        )

    @field_validator(
        "TRITON_BATCH_TARGETS", "TRITON_BATCH_WAIT_MS", mode="before"
    )
    @classmethod
    def _reject_boolean_batch_map_values(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        if isinstance(value, dict):
            boolean_keys = sorted(
                str(key) for key, item in value.items() if isinstance(item, bool)
            )
            if boolean_keys:
                keys = ", ".join(boolean_keys)
                raise ValueError(
                    f"{info.field_name} value(s) for {keys} must be numeric, "
                    "not boolean"
                )
        return value

    @model_validator(mode="after")
    def _validate_triton_settings(self) -> "Settings":
        if self.TRITON_SYSTEM_SHM_SLOTS * self.TRITON_SYSTEM_SHM_SLOT_BYTES > 512 * 1024 * 1024:
            raise ValueError("Triton system shared memory must not exceed 512 MiB")
        unknown_keys = (
            set(self.TRITON_BATCH_TARGETS) | set(self.TRITON_BATCH_WAIT_MS)
        ) - TRITON_MODEL_KEYS
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"unknown Triton model key(s): {unknown}")

        if self.TRITON_BATCH_TARGET_DEFAULT < 1:
            raise ValueError("TRITON_BATCH_TARGET_DEFAULT must be at least 1")
        if not isfinite(self.TRITON_BATCH_WAIT_MS_DEFAULT):
            raise ValueError("TRITON_BATCH_WAIT_MS_DEFAULT must be finite")
        if self.TRITON_BATCH_WAIT_MS_DEFAULT < 0:
            raise ValueError("TRITON_BATCH_WAIT_MS_DEFAULT must be non-negative")
        for name, timeout in (
            ("TRITON_QUEUE_TIMEOUT_SECONDS", self.TRITON_QUEUE_TIMEOUT_SECONDS),
            (
                "TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS",
                self.TRITON_CLIENT_ACQUIRE_TIMEOUT_SECONDS,
            ),
            ("TRITON_INFER_TIMEOUT_SECONDS", self.TRITON_INFER_TIMEOUT_SECONDS),
            ("TRITON_CONFIG_TIMEOUT_SECONDS", self.TRITON_CONFIG_TIMEOUT_SECONDS),
        ):
            if not isfinite(timeout):
                raise ValueError(f"{name} must be finite")
            if timeout <= 0:
                raise ValueError(f"{name} must be positive")
        for name, count in (
            ("ALOG_WORKER_COUNT", self.ALOG_WORKER_COUNT),
            ("TRITON_CLIENT_POOL_SIZE", self.TRITON_CLIENT_POOL_SIZE),
            ("TRITON_EXECUTOR_CONCURRENCY", self.TRITON_EXECUTOR_CONCURRENCY),
            (
                "TRITON_EXECUTOR_QUEUE_CAPACITY",
                self.TRITON_EXECUTOR_QUEUE_CAPACITY,
            ),
        ):
            if count < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.TRITON_EXECUTOR_CONCURRENCY > self.TRITON_CLIENT_POOL_SIZE:
            raise ValueError(
                "TRITON_EXECUTOR_CONCURRENCY must not exceed "
                "TRITON_CLIENT_POOL_SIZE"
            )
        if self.TRITON_TRANSIENT_RETRIES not in (0, 1):
            raise ValueError("TRITON_TRANSIENT_RETRIES must be 0 or 1")

        for model_key in TRITON_MODEL_KEYS:
            target = self.TRITON_BATCH_TARGETS.get(
                model_key, self.TRITON_BATCH_TARGET_DEFAULT
            )
            wait_ms = self.TRITON_BATCH_WAIT_MS.get(
                model_key, self.TRITON_BATCH_WAIT_MS_DEFAULT
            )
            if target < 1:
                raise ValueError(f"{model_key} batch target must be at least 1")
            if not isfinite(wait_ms):
                raise ValueError(f"{model_key} batch wait must be finite")
            if wait_ms < 0:
                raise ValueError(f"{model_key} batch wait must be non-negative")
            if target > 1 and wait_ms <= 0:
                raise ValueError(
                    f"{model_key} batch wait must be positive when target exceeds 1"
                )
            if wait_ms / 1000.0 >= self.TRITON_QUEUE_TIMEOUT_SECONDS:
                raise ValueError(
                    f"{model_key} batch wait must be shorter than "
                    "TRITON_QUEUE_TIMEOUT_SECONDS"
                )

        return self

    @property
    def kafka_bootstrap_servers(self) -> str:
        return f"{self.KAFKA_BROKER_ADDR}:{self.KAFKA_BROKER_PORT}"

    @property
    def kafka_client_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "bootstrap_servers": self.kafka_bootstrap_servers,
            "security_protocol": self.KAFKA_SECURITY_PROTOCOL,
        }

        if self.KAFKA_SECURITY_PROTOCOL.upper().startswith("SASL"):
            if not self.KAFKA_USERNAME or not self.KAFKA_PASSWORD:
                raise RuntimeError(
                    "Kafka SASL protocol requires KAFKA_USERNAME and KAFKA_PASSWORD"
                )

            config.update(
                {
                    "sasl_mechanism": self.KAFKA_SASL_MECHANISM,
                    "sasl_plain_username": self.KAFKA_USERNAME,
                    "sasl_plain_password": self.KAFKA_PASSWORD,
                }
            )

        return config

    @property
    def postgres_dsn(self) -> str:
        user = quote_plus(self.PGSQL_USER)
        password = quote_plus(self.PGSQL_PSWD)
        database = quote_plus(self.PGSQL_DB)
        return (
            f"postgresql+asyncpg://{user}:{password}"
            f"@{self.PGSQL_HOST}:{self.PGSQL_PORT}/{database}"
        )

    @property
    def milvus_uri(self) -> str:
        return self.MILVUS_URI

    @property
    def milvus_token(self) -> str:
        return self.MILVUS_TOKEN


def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        details = []
        for error in exc.errors(include_url=False, include_input=False):
            field = ".".join(str(part) for part in error.get("loc", ()))
            message = error.get("msg", "Invalid value")
            details.append(f"{field}: {message}" if field else message)
        joined = "; ".join(details)
        raise RuntimeError(
            f"Invalid ALog configuration. Missing or invalid variables: {joined}"
        ) from None


settings = _load_settings()
