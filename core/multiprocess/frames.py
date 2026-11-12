"""视觉端仅持有远端帧租约；全局容量与租约状态由owner持有。"""
from contextlib import asynccontextmanager
import asyncio
from core.admission import FrameLease


class RemoteFrameBudget:
    def __init__(self, peer):
        self.peer = peer
        self.inflight = self.waiting = 0
        self._leases = set()

    @asynccontextmanager
    async def frame(self):
        self.waiting += 1
        acquiring = asyncio.create_task(self.peer.call("frame.acquire", {}))
        try:
            lease_id = await asyncio.shield(acquiring)
        except asyncio.CancelledError:
            # 已发送请求仍可能得到真实租约；等到回包后归还，不能遗失credit。
            while not acquiring.done():
                try:
                    await asyncio.shield(acquiring)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not acquiring.cancelled() and acquiring.exception() is None:
                cleanup = asyncio.create_task(self.peer.call("frame.release", {
                    "lease_id": acquiring.result(), "observed_image_bytes": 0,
                }))
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if not cleanup.cancelled():
                    cleanup.exception()
            raise
        finally:
            self.waiting -= 1
        self.inflight += 1
        lease = FrameLease()
        self._leases.add(lease)
        try:
            yield lease
        finally:
            try:
                await self.peer.call("frame.release", {
                    "lease_id": lease_id, "observed_image_bytes": lease.image_bytes,
                })
            finally:
                self._leases.discard(lease)
                self.inflight -= 1

    def queue_snapshot(self):
        return {"inflight_frames": self.inflight, "waiting_frames": self.waiting,
                "observed_image_bytes": sum(lease.image_bytes for lease in self._leases),
                "authority": "owner"}
