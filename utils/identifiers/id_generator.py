import time
import threading
from typing import Optional
from loguru import logger

class SnowflakeGenerator:
    """
    雪花ID生成器实现
    结构: 1位符号位 + 41位时间戳 + 10位机器ID + 12位序列号
    """
    
    def __init__(self, datacenter_id: int, worker_id: int, epoch: int = 1672531200000):
        """
        初始化ID生成器
        
        :param datacenter_id: 数据中心ID (0-31)
        :param worker_id: 工作机器ID (0-31) 
        :param epoch: 自定义起始时间戳(毫秒)
        """
        if not (0 <= datacenter_id <= 31):
            raise ValueError("datacenter_id必须在0-31之间")
        if not (0 <= worker_id <= 31):
            raise ValueError("worker_id必须在0-31之间")
            
        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.epoch = epoch
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()
        
    def _current_millis(self) -> int:
        """获取当前时间戳(毫秒)"""
        return int(time.time() * 1000)
        
    def _wait_next_millis(self, last_timestamp: int) -> int:
        """等待下一毫秒"""
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp
        
    def generate(self) -> Optional[int]:
        """
        生成雪花ID
        
        :return: 64位整数ID
        :raises: Exception 当时钟回拨时抛出异常
        """
        with self.lock:
            timestamp = self._current_millis()
            
            # 处理时钟回拨
            if timestamp < self.last_timestamp:
                logger.error(f"时钟回拨 detected: {self.last_timestamp - timestamp}ms")
                raise Exception(f"时钟回拨 detected. 拒绝生成ID直到 {self.last_timestamp - timestamp}ms")
                
            # 同一毫秒内生成
            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & 0xFFF
                if self.sequence == 0:  # 当前毫秒序列号用完
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                self.sequence = 0
                
            self.last_timestamp = timestamp
            
            return ((timestamp - self.epoch) << 22) | \
                   (self.datacenter_id << 17) | \
                   (self.worker_id << 12) | \
                   self.sequence

# 全局ID生成器实例(默认数据中心0，工作机器0)
id_generator = SnowflakeGenerator(datacenter_id=0, worker_id=0)
