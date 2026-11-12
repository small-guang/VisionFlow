import time
from typing import Dict, List, Optional, Tuple

from loguru import logger
from sqlalchemy.future import select

from db.pgsql.cam_device import SessionLocal
from db.pgsql.device_cluster import DeviceCluster


CACHE_EXPIRE_SECONDS = 7 * 24 * 60 * 60


class ClusterDeviceCache:
    def __init__(self, expire_seconds: int = CACHE_EXPIRE_SECONDS):
        self.expire_seconds = expire_seconds
        self._cache: Dict[str, Tuple[int, float]] = {}

    async def get_cluster_devices(self, device_sn: str) -> Optional[List[str]]:
        if not device_sn:
            return None

        if device_sn in self._cache:
            cluster_id, cached_at = self._cache[device_sn]
            if time.time() - cached_at < self.expire_seconds:
                try:
                    return await self._load_devices_by_cluster_id(cluster_id)
                except Exception as e:
                    logger.error(f"Failed to get cluster devices from cache: {e}")

        try:
            async with SessionLocal() as session:
                result = await session.execute(
                    select(DeviceCluster).filter(
                        DeviceCluster.device_sn == device_sn,
                        DeviceCluster.deleted_at.is_(None),
                    )
                )
                device = result.scalars().first()
                if not device:
                    logger.error(f"Device {device_sn} has no cluster information")
                    return None

                self._cache[device_sn] = (device.cluster_id, time.time())
                result = await session.execute(
                    select(DeviceCluster).filter(
                        DeviceCluster.cluster_id == device.cluster_id,
                        DeviceCluster.deleted_at.is_(None),
                    )
                )
                cluster_devices = result.scalars().all()
                return [d.device_sn for d in cluster_devices]
        except Exception as e:
            logger.error(f"Failed to get cluster devices: {e}")
            return None

    async def _load_devices_by_cluster_id(self, cluster_id: int) -> List[str]:
        async with SessionLocal() as session:
            result = await session.execute(
                select(DeviceCluster).filter(
                    DeviceCluster.cluster_id == cluster_id,
                    DeviceCluster.deleted_at.is_(None),
                )
            )
            devices = result.scalars().all()
            return [d.device_sn for d in devices]
