"""I/O 观测结果的计数、窗口和日志完整性契约；不访问真实服务。"""
import json
from io import StringIO
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from performance_tools.observe_io import IOObserver, JsonlTail, disk_delta, main, parse_diskstats, parse_io_stat, pipeline_window


class DiskCounterTests(unittest.TestCase):
    def test_rates_keep_flush_separate_and_allow_inflight_to_fall(self):
        before = parse_diskstats("8 0 sda 100 0 2048 200 200 0 4096 800 7 1000 5000 0 0 0 0 20 400")['sda']
        after = parse_diskstats("8 0 sda 110 0 6144 240 220 0 12288 1000 0 1500 6000 0 0 0 0 25 500")['sda']
        result = disk_delta(before, after, 2)
        self.assertTrue(result['valid'])
        self.assertEqual(result['read_iops'], 5)
        self.assertEqual(result['write_iops'], 10)
        self.assertEqual(result['read_mib_s'], 1)
        self.assertEqual(result['write_mib_s'], 2)
        self.assertEqual(result['write_await_ms'], 10)
        self.assertEqual(result['flush_per_s'], 2.5)
        self.assertEqual(result['flush_await_ms'], 20)
        self.assertEqual(result['average_queue'], 0.5)
        self.assertEqual(result['util_percent'], 25)

    def test_reset_and_device_replacement_do_not_produce_negative_rates(self):
        before = parse_diskstats("8 0 sda 10 0 100 20 30 0 200 40 1 50 60")['sda']
        after = {**before, 'writes': 2}
        self.assertEqual(disk_delta(before, after, 1)['reason'], 'counter_reset')
        self.assertEqual(disk_delta(before, {**before, 'minor': 1}, 1)['reason'], 'device_identity_changed')
        for seconds in [0, -1, float('nan'), float('inf')]:
            self.assertFalse(disk_delta(before, before, seconds)['valid'])

    def test_older_kernel_and_idle_window_have_no_invented_flush_latency(self):
        text = "8 0 sda 10 0 100 20 30 0 200 40 0 50 60\n179 0 mmcblk0 invalid\n"
        devices = parse_diskstats(text, {'sda', 'mmcblk0'})
        self.assertEqual(set(devices), {'sda'})
        result = disk_delta(devices['sda'], devices['sda'], 1)
        self.assertTrue(result['valid'])
        self.assertEqual(result['write_iops'], 0)
        self.assertIsNone(result['write_await_ms'])
        self.assertIsNone(result['flush_per_s'])
        self.assertIsNone(result['flush_await_ms'])

    def test_cgroup_unknown_device_and_optional_counters_are_preserved(self):
        result = parse_io_stat("8:0 rbytes=1024 wbytes=2048 rios=1 wios=2 dbytes=0\n179:0 rbytes=3 wbytes=4\nbad input\n")
        self.assertEqual(result['8:0']['wbytes'], 2048)
        self.assertEqual(result['179:0']['rbytes'], 3)
        self.assertNotIn('bad', result)

    def test_counter_schema_and_invalid_values_cannot_silently_become_rates(self):
        row = parse_diskstats("8 0 sda 10 0 100 20 30 0 200 40 0 50 60")['sda']
        self.assertEqual(disk_delta(row, {**row, 'flushes': 1, 'flush_ms': 5}, 1)['reason'], 'counter_schema_changed')
        for value in [None, True, -1, 3.5, float('nan')]:
            self.assertFalse(disk_delta({**row, 'writes': value}, row, 1)['valid'])
        missing_identity = {k: v for k, v in row.items() if k != 'major'}
        self.assertFalse(disk_delta(missing_identity, missing_identity, 1)['valid'])


class PipelineWindowTests(unittest.TestCase):
    @staticmethod
    def snapshot(timestamp, images=0, messages=0, session='one'):
        return {'timestamp': timestamp, 'session_id': session, 'pid': 1,
                'counters': {'image_pipeline_completed': {'total': images}},
                'stages': {'message_processing': {'completed_total': messages}}}

    def test_counts_use_real_window_not_nominal_five_second_interval(self):
        before = self.snapshot('2026-10-20T07:00:00+00:00', 100, 90)
        after = self.snapshot('2026-10-20T07:00:12+00:00', 220, 198)
        result = pipeline_window(before, after)
        self.assertEqual(result['image_completed_per_second'], 10)
        self.assertEqual(result['message_completed_per_second'], 9)
        self.assertTrue(result['image_counter_may_include_pano'])
        self.assertTrue(result['not_output_persistence_completion'])

    def test_session_change_reset_and_bad_timestamp_cannot_be_combined(self):
        before = self.snapshot('2026-10-20T07:00:00+00:00', 100)
        after = self.snapshot('2026-10-20T07:00:05+00:00', 120, session='two')
        self.assertEqual(pipeline_window(before, after)['reason'], 'session_changed')
        after['session_id'] = 'one'
        after['counters']['image_pipeline_completed']['total'] = 1
        self.assertEqual(pipeline_window(before, after)['reason'], 'counter_reset')
        after['timestamp'] = 'invalid'
        self.assertFalse(pipeline_window(before, after)['valid'])
        self.assertFalse(pipeline_window(before, before)['valid'])

    def test_missing_counter_is_unknown_even_when_the_other_endpoint_is_valid(self):
        before = self.snapshot('2026-10-20T07:00:00+00:00', 100, 90)
        after = self.snapshot('2026-10-20T07:00:05+00:00', 120, 110)
        for side in [0, 1]:
            for path in [('counters', 'image_pipeline_completed', 'total'), ('stages', 'message_processing', 'completed_total')]:
                pair = json.loads(json.dumps([before, after]))
                del pair[side][path[0]][path[1]][path[2]]
                result = pipeline_window(*pair)
                self.assertFalse(result['valid'])
                self.assertEqual(result['reason'], 'missing_counters')
                self.assertNotIn('image_completed_per_second', result)

    def test_backwards_and_naive_time_and_missing_pid_are_rejected(self):
        before = self.snapshot('2026-10-20T07:00:00+00:00', 100, 90)
        after = self.snapshot('2026-10-20T07:00:05+00:00', 120, 110)
        self.assertFalse(pipeline_window(after, before)['valid'])
        after['timestamp'] = '2026-10-20T07:00:05'
        self.assertFalse(pipeline_window(before, after)['valid'])
        after['timestamp'] = '2026-10-20T07:00:05+00:00'
        del before['pid']
        del after['pid']
        self.assertFalse(pipeline_window(before, after)['valid'])

    def test_explicit_zero_counters_are_valid_and_not_confused_with_missing(self):
        before = self.snapshot('2026-10-20T07:00:00+00:00')
        after = self.snapshot('2026-10-20T07:00:05+00:00')
        result = pipeline_window(before, after)
        self.assertTrue(result['valid'])
        self.assertEqual(result['image_completed_per_second'], 0)
        self.assertEqual(result['start_timestamp'], before['timestamp'])
        self.assertEqual(result['end_timestamp'], after['timestamp'])


class ObserverBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.proc = self.base / 'proc'
        self.group = self.base / 'cgroup'
        (self.proc / '123').mkdir(parents=True)
        self.container = self.group / 'system.slice/docker-one.scope'
        self.container.mkdir(parents=True)
        self.write_stat(100)
        (self.proc / '123/io').write_text('read_bytes: 1\nwrite_bytes: 2\n')
        (self.proc / '123/cgroup').write_text('0::/system.slice/docker-one.scope\n')
        (self.proc / 'stat').write_text('cpu 1 2 3 4 5 6 7 8\n')
        (self.proc / 'diskstats').write_text('8 0 sda 10 0 100 20 30 0 200 40 0 50 60\n')
        (self.proc / 'vmstat').write_text('pswpin 1\npswpout 2\n')
        (self.proc / 'meminfo').write_text('MemAvailable: 100 kB\n')
        (self.container / 'io.stat').write_text('8:0 rbytes=1 wbytes=2 rios=1 wios=2\n')

    def write_stat(self, start_ticks):
        fields = ['0'] * 22
        fields[0] = 'S'
        fields[19] = str(start_ticks)
        (self.proc / '123/stat').write_text('123 (name with parentheses (x)) ' + ' '.join(fields))

    def observer(self):
        return IOObserver({'alog': 123}, proc=self.proc, cgroup=self.group, sysfs=self.base / 'sys')

    def test_permission_denied_io_preserves_process_and_marks_unavailable(self):
        observer = self.observer()
        original = Path.read_text
        denied = {self.proc / '123/io', self.container / 'io.stat'}

        def read(path, *args, **kwargs):
            if path in denied:
                raise PermissionError(13, 'denied')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', read):
            result = observer.sample(time.monotonic())
        process = result['processes'][0]
        self.assertEqual(process['start_ticks'], 100)
        self.assertFalse(process['io_available'])
        self.assertNotIn('io', process)
        group = next(g for g in result['cgroups'] if g['path'] == str(self.container))
        self.assertFalse(group['availability']['io.stat'])
        self.assertNotIn('io.stat', group)
        failures = [e for e in result['errors'] if e['type'] == 'PermissionError']
        self.assertEqual(len(failures), 2)

    def test_reused_pid_does_not_become_the_original_service(self):
        observer = self.observer()
        self.write_stat(200)
        result = observer.sample(time.monotonic())
        self.assertEqual(result['processes'], [])
        self.assertTrue(any(e['type'] == 'PidReused' for e in result['errors']))

    def test_summary_uses_sample_bounds_not_collection_lifecycle_for_windows(self):
        clock = [100.0]

        class FakeObserver:
            root_identities = {}
            errors = []

            def __init__(self, roots):
                pass

            def sample(self, now):
                clock[0] += 0.2
                return {'timestamp': now + 1000, 'monotonic': now, 'disks': {},
                        'diskstats_available': True, 'disk_monotonic': now + 0.1}

        def sleep(seconds):
            clock[0] += seconds

        output = self.base / 'output'
        with patch('performance_tools.observe_io.IOObserver', FakeObserver), \
             patch('performance_tools.observe_io.time.monotonic', lambda: clock[0]), \
             patch('performance_tools.observe_io.time.sleep', sleep), \
             patch('performance_tools.observe_io.os.sysconf', lambda name: 4096 if name == 'SC_PAGE_SIZE' else 100, create=True), \
             patch('sys.stdout', new_callable=StringIO):
            self.assertEqual(main(['--output', str(output), '--seconds', '2', '--interval', '1',
                                   '--pipeline-path', str(self.base / 'missing.jsonl')]), 0)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['resource_samples'], 3)
        self.assertAlmostEqual(summary['elapsed_seconds'], 2.2)
        self.assertAlmostEqual(summary['resource_window_seconds'], 2)
        self.assertAlmostEqual(summary['disk_window_seconds'], 2)
        self.assertEqual(summary['pipeline_windows'], {})


class JsonlTailTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'pipeline.jsonl'

    def test_initial_latest_partial_completion_and_no_duplicate(self):
        self.path.write_bytes(b'{"n":1}\n{"n":2}\n{"n":')
        tail = JsonlTail(self.path)
        rows, status = tail.poll()
        self.assertEqual(rows, [{'n': 2}])
        self.assertGreater(status['partial_bytes'], 0)
        with self.path.open('ab') as stream:
            stream.write(b'3}\n{"n":4}\n')
        self.assertEqual(tail.poll()[0], [{'n': 3}, {'n': 4}])
        self.assertEqual(tail.poll()[0], [])

    def test_rotation_drops_old_partial_and_reads_new_inode(self):
        self.path.write_bytes(b'{"n":1}\n{"old":')
        tail = JsonlTail(self.path)
        tail.poll()
        self.path.rename(self.path.with_suffix('.old'))
        self.path.write_bytes(b'{"n":2}\n')
        rows, status = tail.poll()
        self.assertEqual(rows, [{'n': 2}])
        self.assertEqual(status['reset'], 'rotation_or_rewrite')

    def test_copytruncate_with_file_regrown_past_offset_is_detected(self):
        self.path.write_bytes(b'{"n":1}\n')
        tail = JsonlTail(self.path)
        tail.poll()
        self.path.write_bytes(b'{"n":900}\n{"n":901}\n')
        rows, status = tail.poll()
        self.assertEqual(rows, [{'n': 900}, {'n': 901}])
        self.assertEqual(status['reset'], 'rotation_or_rewrite')

    def test_missing_file_and_bad_lines_do_not_stop_observation(self):
        tail = JsonlTail(self.path)
        rows, status = tail.poll()
        self.assertEqual(rows, [])
        self.assertEqual(status['error']['type'], 'FileNotFoundError')
        self.path.write_bytes(b'not json\n[1,2]\n{"n":7}\n')
        rows, status = tail.poll()
        self.assertEqual(rows, [{'n': 7}])
        self.assertEqual(status['invalid_lines'], 2)

    def test_bounded_catchup_reports_skipped_data(self):
        self.path.write_bytes(b'{"n":0}\n')
        tail = JsonlTail(self.path, max_bytes=64)
        tail.poll()
        with self.path.open('ab') as stream:
            for number in range(1, 50):
                stream.write((json.dumps({'n': number}) + '\n').encode())
        rows, status = tail.poll()
        self.assertEqual(status['gap'], 'tail_limit_exceeded')
        self.assertEqual(rows[-1], {'n': 49})
        self.assertLess(len(rows), 49)


if __name__ == '__main__':
    unittest.main()
