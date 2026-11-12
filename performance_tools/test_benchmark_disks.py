"""额外设备在 /proc/diskstats 存在、缺失和默认模式下的采样契约。"""
import tempfile
import time
import unittest
from pathlib import Path

from performance_tools.observe_io import IOObserver, disk_delta


class AdditionalDiskTests(unittest.TestCase):
    def test_additional_device_and_missing_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / 'proc'
            proc.mkdir()
            (proc / 'diskstats').write_text(
                '8 0 sda 10 0 100 20 30 0 200 40 0 50 60\n'
                '259 0 nvme0n1 10 0 100 20 30 0 200 40 0 50 60\n')
            observer = IOObserver({}, proc=proc, cgroup=root / 'cgroup', sysfs=root / 'sys',
                                  disk_devices={'sda', 'nvme0n1', 'mmcblk0'})
            before = observer.sample(time.monotonic())
            self.assertEqual(set(before['disks']), {'sda', 'nvme0n1'})
            self.assertEqual(before['missing_disk_devices'], ['mmcblk0'])
            (proc / 'diskstats').write_text(
                '8 0 sda 10 0 100 20 30 0 200 40 0 50 60\n'
                '259 0 nvme0n1 12 0 104 24 34 0 208 52 0 55 70\n')
            after = observer.sample(time.monotonic())
            delta = disk_delta(before['disks']['nvme0n1'], after['disks']['nvme0n1'], 2)
            self.assertTrue(delta['valid'])
            self.assertEqual(delta['write_iops'], 2)
            self.assertEqual(delta['write_await_ms'], 3)
            default = IOObserver({}, proc=proc, cgroup=root / 'cgroup', sysfs=root / 'sys').sample(time.monotonic())
            self.assertEqual(set(default['disks']), {'sda'})
            self.assertEqual(default['missing_disk_devices'], ['mmcblk0'])


if __name__ == '__main__':
    unittest.main()
