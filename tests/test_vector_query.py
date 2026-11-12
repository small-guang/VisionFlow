"""跨请求查询的兼容性、背压、取消和停机；不连接 Milvus。"""

import asyncio
from copy import deepcopy
import threading
from types import SimpleNamespace
import unittest

import numpy as np

from core.execution import AsyncWorkExecutor, ExecutionResources, OwnerExecutionResources
from core.vector_query import VectorQueryCoordinator
from core.performance import RuntimePerformance
from tests.test_execution_resources import wait_thread_event
from tests.test_face_gallery import load_gallery_modules


PARAMETERS = {
    "collection_name": "face", "anns_field": "embedding", "filter": "",
    "search_params": {"metric_type": "IP", "params": {"nprobe": 10}},
    "limit": 100, "output_fields": ["faceId", "sampleId"],
    "consistency_level": "Strong",
}


class Client:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return [[{"entity": {"faceId": str(vector[0]), "sampleId": f"sample-{vector[0]}"},
                  "distance": vector[0]}] for vector in kwargs["data"]]


class VectorQueryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = AsyncWorkExecutor(2, name="vector_search")
        self.addAsyncCleanup(self.executor.close)

    def coordinator(self, **kwargs):
        coordinator = VectorQueryCoordinator(self.executor, **kwargs)
        self.addAsyncCleanup(coordinator.close)
        return coordinator

    async def test_combines_queries_and_restores_each_result_in_order(self):
        client, coordinator = Client(), self.coordinator(batch_size=4)
        results = await asyncio.gather(*(
            coordinator.search(client, [float(index), 1.0], **PARAMETERS) for index in range(9)
        ))
        self.assertEqual([hits[0]["distance"] for hits in results], list(range(9)))
        self.assertEqual([len(call["data"]) for call in client.calls], [4, 4, 1])
        for call in client.calls:
            self.assertEqual({key: value for key, value in call.items() if key != "data"}, PARAMETERS)
        snapshot = coordinator.queue_snapshot()
        self.assertEqual(snapshot["finished_rpcs"], 3)
        self.assertEqual(snapshot["finished_rpc_queries"], 9)
        self.assertEqual(snapshot["batch_histogram"], {4: 2, 1: 1})
        self.assertEqual(snapshot["admitted_queries"], 0)
        self.assertGreater(snapshot["batch_wait_total_ms"], 0)
        self.assertGreater(snapshot["rpc_run_total_ms"], 0)

    async def test_every_search_parameter_client_and_dimension_partition_batches(self):
        client, second, coordinator = Client(), Client(), self.coordinator(batch_size=32, wait_ms=1)
        changes = [
            {"collection_name": "other"}, {"anns_field": "other"}, {"filter": "deviceId == 1"},
            {"search_params": {"metric_type": "COSINE", "params": {"nprobe": 10}}},
            {"search_params": {"metric_type": "IP", "params": {"nprobe": 11}}},
            {"limit": 101}, {"output_fields": ["faceId"]}, {"consistency_level": "Bounded"},
            {"partition_names": ["p"]}, {"timeout": 2.0}, {"group_by_field": "faceId"},
            {"group_size": 3}, {"strict_group_size": True},
        ]
        requests = [coordinator.search(client, [1.0, 2.0], **(PARAMETERS | change)) for change in changes]
        requests.extend((coordinator.search(client, [1.0], **PARAMETERS),
                         coordinator.search(second, [1.0, 2.0], **PARAMETERS),
                         coordinator.search(client, [1.0, 2.0], **PARAMETERS)))
        await asyncio.gather(*requests)
        self.assertEqual(len(client.calls), len(changes) + 2)
        self.assertEqual(len(second.calls), 1)
        self.assertTrue(all(len(call["data"]) == 1 for call in client.calls + second.calls))

    async def test_parameter_snapshot_and_nested_key_order(self):
        client, coordinator = Client(), self.coordinator(batch_size=2, wait_ms=10)
        parameters = deepcopy(PARAMETERS)
        task = asyncio.create_task(coordinator.search(client, [1.0], **parameters))
        await asyncio.sleep(0)
        parameters["search_params"]["params"]["nprobe"] = 99
        reordered = dict(reversed(list(PARAMETERS.items())))
        result = await coordinator.search(client, [2.0], **reordered)
        await task
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["search_params"]["params"]["nprobe"], 10)
        self.assertEqual(result[0]["distance"], 2)

    async def test_cancel_before_dispatch_removes_query_and_empty_compatibility_group(self):
        client, coordinator = Client(), self.coordinator(batch_size=4, wait_ms=10)
        task = asyncio.create_task(coordinator.search(client, [1.0], **PARAMETERS))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await coordinator.close()
        self.assertEqual(client.calls, [])
        self.assertEqual(coordinator.queue_snapshot()["admitted_queries"], 0)
        self.assertEqual(coordinator.queue_snapshot()["compatible_groups"], 0)

    async def test_inflight_cancel_retains_capacity_and_close_drains_real_sdk(self):
        entered, release = threading.Event(), threading.Event()
        client, coordinator = Client(), self.coordinator(batch_size=2, capacity=2)
        original = client.search

        def block(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release SDK")
            return original(**kwargs)

        client.search = block
        queries = [asyncio.create_task(coordinator.search(client, [index], **PARAMETERS)) for index in (1, 2)]
        closing = waiting = None
        try:
            await wait_thread_event(entered)
            waiting = asyncio.create_task(coordinator.search(client, [3], **PARAMETERS))
            for query in queries:
                query.cancel()
            await asyncio.gather(*queries, return_exceptions=True)
            await asyncio.sleep(0)
            self.assertEqual(coordinator.queue_snapshot()["admitted_queries"], 2)
            self.assertEqual(coordinator.queue_snapshot()["admission_waiters"], 1)
            closing = asyncio.create_task(coordinator.close())
            await asyncio.sleep(0)
            closing.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await waiting
            self.assertFalse(coordinator._close_task.done())
            # 同一 close 生命周期的第二个调用者取消仍不能中断排空。
            closing_again = asyncio.create_task(coordinator.close())
            await asyncio.sleep(0)
            closing_again.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await closing_again
            self.assertFalse(coordinator._close_task.done())
            self.assertEqual(coordinator.queue_snapshot()["admitted_queries"], 2)
            release.set()
            await coordinator.close()
            self.assertEqual(coordinator.queue_snapshot()["admitted_queries"], 0)
            self.assertEqual(len(client.calls), 1)
        finally:
            release.set()
            await asyncio.gather(*(task for task in [*queries, closing, waiting] if task), return_exceptions=True)
            await coordinator.close()

    async def test_one_cancelled_query_does_not_cancel_other_batch_result(self):
        entered, release = threading.Event(), threading.Event()
        client, coordinator = Client(), self.coordinator(batch_size=2)
        original = client.search

        def block(**kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test did not release SDK")
            return original(**kwargs)

        client.search = block
        first = asyncio.create_task(coordinator.search(client, [1.0], **PARAMETERS))
        second = asyncio.create_task(coordinator.search(client, [2.0], **PARAMETERS))
        try:
            await wait_thread_event(entered)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            self.assertEqual((await second)[0]["distance"], 2)
        finally:
            release.set()
            await asyncio.gather(first, second, return_exceptions=True)

    async def test_close_flushes_accepted_partial_batch_and_rejects_new_queries(self):
        client, coordinator = Client(), self.coordinator(batch_size=4, wait_ms=10)
        task = asyncio.create_task(coordinator.search(client, [1.0], **PARAMETERS))
        await asyncio.sleep(0)
        await coordinator.close()
        self.assertEqual((await task)[0]["distance"], 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await coordinator.search(client, [2.0], **PARAMETERS)

    async def test_malformed_response_fails_whole_group_without_truncating(self):
        for results in ([], [[]], [[], [], []], [{}, []], [[None], []]):
            with self.subTest(results=results):
                coordinator = self.coordinator(batch_size=2)
                client = SimpleNamespace(search=lambda **kwargs: results)
                outcomes = await asyncio.gather(*(
                    coordinator.search(client, [index], **PARAMETERS) for index in (1, 2)
                ), return_exceptions=True)
                self.assertTrue(all(isinstance(outcome, ValueError) for outcome in outcomes))
                self.assertEqual(coordinator.queue_snapshot()["failed_queries"], 2)
                self.assertEqual(coordinator.queue_snapshot()["admitted_queries"], 0)

    async def test_empty_hits_and_original_sdk_failure_are_distributed(self):
        coordinator = self.coordinator(batch_size=2)
        client = SimpleNamespace(search=lambda **kwargs: [[] for _ in kwargs["data"]])
        self.assertEqual(await asyncio.gather(*(
            coordinator.search(client, [index], **PARAMETERS) for index in (1, 2)
        )), [[], []])
        failure = RuntimeError("remote search failed")

        def fail(**kwargs):
            raise failure

        client.search = fail
        outcomes = await asyncio.gather(*(
            coordinator.search(client, [index], **PARAMETERS) for index in (1, 2)
        ), return_exceptions=True)
        self.assertTrue(all(outcome is failure for outcome in outcomes))

    async def test_unrecognized_parameters_are_rejected_before_admission(self):
        client, coordinator = Client(), self.coordinator()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            await coordinator.search(client, [1.0], ranker=object(), **PARAMETERS)
        self.assertEqual(coordinator.queue_snapshot()["admitted_queries"], 0)
        self.assertEqual(client.calls, [])

    async def test_installed_milvus_result_types_and_performance_metrics(self):
        from pymilvus.client.abstract import SearchResult
        from pymilvus.grpc_gen import schema_pb2

        response = SearchResult(schema_pb2.SearchResultData(
            num_queries=2, top_k=1, topks=[1, 1], scores=[0.9, 0.8],
            ids=schema_pb2.IDs(int_id=schema_pb2.LongArray(data=[11, 22])),
        ))
        client = SimpleNamespace(search=lambda **kwargs: response)
        metrics = RuntimePerformance()
        coordinator = self.coordinator(batch_size=2, metrics=metrics)
        actual = await asyncio.gather(*(
            coordinator.search(client, [index], **PARAMETERS) for index in (1, 2)
        ))
        self.assertEqual([hits[0]["id"] for hits in actual], [11, 22])
        snapshot = metrics.snapshot()["operations"]
        self.assertEqual(snapshot["vector_query_admission_wait"]["count"], 2)
        self.assertEqual(snapshot["vector_query_batch_wait"]["count"], 2)
        self.assertEqual(snapshot["vector_query_rpc_run"]["count"], 1)

    async def test_owner_reuses_existing_pool_and_drains_before_pool_close(self):
        owner = OwnerExecutionResources(search_workers=1, write_workers=1, query_batch_size=4)
        try:
            self.assertIs(owner.vector_queries._executor, owner.vector_search)
            self.assertEqual(set(owner._executors), {"vector_search", "vector_write"})
            task = asyncio.create_task(owner.vector_queries.search(Client(), [1.0], **PARAMETERS))
            await asyncio.sleep(0)
            await owner.close()
            self.assertEqual((await task)[0]["distance"], 1)
            self.assertTrue(all(item["closed"] for item in owner.queue_snapshot().values()))
        finally:
            await owner.close()


class FaceQueryBatchTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.module = load_gallery_modules()

    async def test_batch_one_uses_the_same_coordinator_for_both_resource_roles(self):
        for resource_type, budgets in (
            (ExecutionResources, {"image_workers": 1, "storage_workers": 1}),
            (OwnerExecutionResources, {}),
        ):
            with self.subTest(role=resource_type.__name__):
                async with resource_type(search_workers=1, write_workers=1,
                                         query_batch_size=1, **budgets) as resources:
                    gallery = object.__new__(self.module.FaceGallery)
                    gallery.client = Client()
                    gallery.collection_name = "face"
                    gallery.query_coordinator = resources.vector_queries
                    result = await gallery.asearch_similar_faces_with_hit_info(np.array([0.95]))
                    self.assertEqual(result, ("0.95", 0.95, "0.95"))
                    snapshot = resources.queue_snapshot()["vector_queries"]
                    self.assertEqual(snapshot["batch_size"], 1)
                    self.assertEqual(snapshot["finished_rpc_queries"], 1)
                    self.assertEqual(snapshot["batch_histogram"], {1: 1})
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    await gallery.asearch_face_samples(np.array([0.95]))

    async def test_async_gallery_queries_fail_without_an_injected_coordinator(self):
        gallery = object.__new__(self.module.FaceGallery)
        gallery.query_coordinator = None
        for operation in (gallery.asearch_face_samples, gallery.asearch_similar_faces_with_hit_info):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(RuntimeError, "coordinator"):
                await operation(np.array([0.95]))

    async def test_gallery_batch_and_single_query_have_identical_samples_and_parameters(self):
        executor = AsyncWorkExecutor(2, name="vector_search")
        coordinator = VectorQueryCoordinator(executor, batch_size=4)
        gallery = object.__new__(self.module.FaceGallery)
        gallery.client = Client()
        gallery.collection_name = "face"
        gallery.search_executor = executor
        gallery.query_coordinator = coordinator
        vectors = [np.array([index / 10, 0.1], dtype=np.float32) for index in range(4)]
        try:
            expected = [gallery.search_face_samples(vector, ["device"]) for vector in vectors]
            old_calls = gallery.client.calls[:]
            gallery.client.calls.clear()
            actual = await asyncio.gather(*(gallery.asearch_face_samples(vector, ["device"]) for vector in vectors))
            self.assertEqual(actual, expected)
            self.assertEqual(len(gallery.client.calls), 1)
            merged = gallery.client.calls[0]
            self.assertEqual(merged["data"], [call["data"][0] for call in old_calls])
            self.assertEqual({key: value for key, value in merged.items() if key != "data"},
                             {key: value for key, value in old_calls[0].items() if key != "data"})
            self.assertEqual(merged["consistency_level"], "Strong")
            self.assertEqual(merged["limit"], 100)
            self.assertEqual(await gallery.asearch_face_samples(np.zeros((2, 2))), [])
        finally:
            await coordinator.close()
            await executor.close()


if __name__ == "__main__":
    unittest.main()
