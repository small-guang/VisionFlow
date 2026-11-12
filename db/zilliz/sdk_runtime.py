"""Milvus SDK 的进程内启动准备，不创建连接或执行查询。"""


def warm_search_dependencies() -> None:
    """在接纳并发搜索前串行初始化 SDK 的可选稀疏类型检查。

    requirements 固定 pymilvus==3.0.1，其 SciPyHelper 首次检查会直接
    exec_module，且没有初始化锁。此处隔离这一 SDK 内部兼容点；升级
    pymilvus 时需复核。调用 SDK 自身入口保留未安装 SciPy 时的行为，
    已安装但损坏的依赖或 SDK 接口变化仍向启动方抛错，不静默跳过。
    """
    from pymilvus.client.utils import SciPyHelper

    SciPyHelper.is_scipy_sparse([])
