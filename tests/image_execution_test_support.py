"""业务规则单测显式注入的执行器；线程隔离由专用资源单测验证。"""


class InlineImageExecutor:
    async def run(self, operation, /, *args, **kwargs):
        return operation(*args, **kwargs)
