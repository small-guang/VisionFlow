"""Milvus依赖工厂；导入包不会连接数据库或创建集合。"""

from __future__ import annotations

from dataclasses import dataclass
import threading

from .face_feature import FaceGallery
from .picture_feature import MilvusVectorDB
from .SBS_feature import SbsReidGallery
from .TBPS_feature import TbpsGallery
from .sdk_runtime import warm_search_dependencies


@dataclass
class DatabaseServices:
    face: FaceGallery
    picture: MilvusVectorDB
    sbs: SbsReidGallery
    tbps: TbpsGallery

    @classmethod
    def create(cls, config, resources) -> DatabaseServices:
        # Runtime 在接纳消息前串行调用此工厂；不把 SDK 首次加载留给搜索池。
        warm_search_dependencies()
        query_coordinator = resources.vector_queries
        if query_coordinator is None:
            raise ValueError("database services require a vector query coordinator")
        created = []
        kwargs = {
            "config": config,
            "search_executor": resources.vector_search,
            "write_executor": resources.vector_write,
        }
        prefix = config.MILVUS_COLLECTION_PREFIX
        try:
            for factory, name in (
                (FaceGallery, "alogFace"), (MilvusVectorDB, "alogPictureClip"),
                (SbsReidGallery, "alogSBS"), (TbpsGallery, "alogTBPS"),
            ):
                extra = ({"query_coordinator": query_coordinator}
                         if factory is FaceGallery else {})
                created.append(factory(collection_name=prefix + name, **kwargs, **extra))
            return cls(*created)
        except BaseException:
            for database in reversed(created):
                try:
                    database.client.close()
                except Exception:
                    pass
            raise

    def close(self) -> None:
        failures = []
        for database in (self.tbps, self.sbs, self.picture, self.face):
            try:
                database.client.close()
            except Exception as error:
                failures.append(error)
        if failures:
            raise RuntimeError("failed to close Milvus clients") from failures[0]


# 兼容旧离线调用；服务入口显式使用DatabaseServices，不经过这些惰性单例。
_legacy = {}
_legacy_lock = threading.RLock()
_legacy_factories = {
    "face_embedding_db": lambda: FaceGallery(collection_name="alogFace", dim=512),
    "person_clip_embedding_db": TbpsGallery,
    "person_sbs_embedding_db": SbsReidGallery,
    "picture_clip_embedding_db": lambda: MilvusVectorDB(collection_name="alogPictureClip", dim=512),
}


def __getattr__(name):
    if name not in _legacy_factories:
        raise AttributeError(name)
    with _legacy_lock:
        if name not in _legacy:
            _legacy[name] = _legacy_factories[name]()
        return _legacy[name]
