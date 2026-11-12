"""当前协程的人脸身份锁与锁内查询观测上下文。"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar


_lock_owner = ContextVar("face_identity_lock_owner", default=None)
_locked_query = ContextVar("face_identity_locked_query", default=False)


@contextmanager
def face_identity_lock_metrics_scope(metrics):
    """只标记持锁协程；异常与取消时恢复原上下文。"""
    token = _lock_owner.set((asyncio.current_task(), metrics))
    try:
        yield
    finally:
        _lock_owner.reset(token)


def current_face_identity_metrics():
    owner = _lock_owner.get()
    if owner is not None and owner[0] is asyncio.current_task():
        return owner[1]
    return None


@contextmanager
def face_identity_locked_query_scope():
    """限定仅对完整人脸查询进行跨批次的持锁归因。"""
    token = _locked_query.set(True)
    try:
        yield
    finally:
        _locked_query.reset(token)


def current_locked_face_query_metrics():
    if _locked_query.get():
        return current_face_identity_metrics()
    return None
