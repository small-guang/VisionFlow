import asyncio
import json
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError

from google.protobuf.json_format import MessageToDict, MessageToJson
from tritonclient.grpc import model_config_pb2

from core.infer.capabilities import (
    ModelCapability,
    discover_all_capabilities,
    discover_model_capability,
)
from core.infer.contracts import MODEL_CONTRACTS
from core.infer.errors import TritonContractError


def _config_datatype(datatype):
    return "TYPE_STRING" if datatype == "BYTES" else f"TYPE_{datatype}"


def model_config_json(contract, max_batch_size):
    return {
        "config": {
            "name": contract.model_name,
            "max_batch_size": max_batch_size,
            "parameters": {
                "api_contract_version": {
                    "string_value": str(contract.api_contract_version)
                },
            },
            "input": [
                {
                    "name": tensor.name,
                    "data_type": _config_datatype(tensor.datatype),
                    "dims": [str(dimension) for dimension in tensor.dims],
                }
                for tensor in contract.inputs
            ],
            "output": [
                {
                    "name": tensor.name,
                    "data_type": _config_datatype(tensor.datatype),
                    "dims": [str(dimension) for dimension in tensor.dims],
                }
                for tensor in contract.outputs
            ],
        }
    }


def model_metadata_json(contract):
    return {
        "name": contract.model_name,
        "inputs": [
            {"name": tensor.name, "datatype": tensor.datatype}
            for tensor in contract.inputs
        ],
        "outputs": [
            {"name": tensor.name, "datatype": tensor.datatype}
            for tensor in contract.outputs
        ],
    }


class FakeContractTransport:
    def __init__(self, contract, *, ready=True, server_max=1):
        self.ready = ready
        self.config = model_config_json(contract, server_max)
        self.metadata = model_metadata_json(contract)
        self.calls = []

    async def is_model_ready(self, model_name):
        self.calls.append(("ready", model_name))
        return self.ready

    async def get_model_config(self, model_name):
        self.calls.append(("config", model_name))
        return self.config

    async def get_model_metadata(self, model_name):
        self.calls.append(("metadata", model_name))
        return self.metadata


class FakeMultiContractTransport:
    def __init__(self, contracts):
        self.ready = {contract.model_name: True for contract in contracts.values()}
        self.configs = {
            contract.model_name: model_config_json(contract, 1)
            for contract in contracts.values()
        }
        self.metadata = {
            contract.model_name: model_metadata_json(contract)
            for contract in contracts.values()
        }

    async def is_model_ready(self, model_name):
        value = self.ready[model_name]
        if isinstance(value, BaseException):
            raise value
        return value

    async def get_model_config(self, model_name):
        return self.configs[model_name]

    async def get_model_metadata(self, model_name):
        return self.metadata[model_name]


class ModelCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_protobuf_json_emits_int64_dims_as_strings(self):
        config = model_config_pb2.ModelConfig(
            name="wire_shape_probe", max_batch_size=1
        )
        tensor = config.input.add(
            name="RAW_IMAGE", data_type=model_config_pb2.TYPE_UINT8
        )
        tensor.dims.extend([-1, -1, 3])

        as_dict = MessageToDict(config, preserving_proto_field_name=True)
        as_json = json.loads(
            MessageToJson(config, preserving_proto_field_name=True)
        )

        self.assertEqual(as_dict["input"][0]["dims"], ["-1", "-1", "3"])
        self.assertEqual(as_json["input"][0]["dims"], ["-1", "-1", "3"])

    async def test_all_contracts_accept_protobuf_string_dims(self):
        for key, contract in MODEL_CONTRACTS.items():
            with self.subTest(key=key):
                capability = await discover_model_capability(
                    FakeContractTransport(contract),
                    contract,
                    target_batch=1,
                )
                self.assertEqual(capability.key, key)

    async def test_person_seg_accepts_packed_mask_crop_contract_at_version_three(self):
        contract = MODEL_CONTRACTS["person_seg"]
        transport = FakeContractTransport(contract, server_max=4)

        capability = await discover_model_capability(
            transport, contract, target_batch=2
        )

        self.assertEqual(
            transport.config["config"]["parameters"]["api_contract_version"]
            ["string_value"],
            "3",
        )
        self.assertEqual(capability.api_contract_version, 3)
        self.assertEqual(capability.server_max_batch, 4)
        self.assertEqual(capability.target_batch, 2)

    async def test_person_seg_rejects_old_full_image_mask_contract_at_version_one(self):
        contract = MODEL_CONTRACTS["person_seg"]
        transport = FakeContractTransport(contract)
        transport.config["config"]["output"] = [
            {
                "name": "DETECTIONS",
                "data_type": "TYPE_FP32",
                "dims": ["100", "6"],
            },
            {
                "name": "MASKS",
                "data_type": "TYPE_UINT8",
                "dims": ["100", "-1", "-1"],
            },
            {
                "name": "NUM_DETECTIONS",
                "data_type": "TYPE_INT32",
                "dims": ["1"],
            },
        ]

        with self.assertRaisesRegex(
            TritonContractError, "person_seg.*MASK_CROPS_PACKED.*MASKS"
        ):
            await discover_model_capability(
                transport, contract, target_batch=1
            )

    async def test_config_dims_reject_non_wire_json_shapes(self):
        contract = MODEL_CONTRACTS["sbs_image"]
        invalid_dims = (
            "-1",
            [-1],
            [True],
            [" "],
            ["+1"],
            ["1.0"],
            ["0x10"],
            ["01"],
            ["-0"],
        )
        for dims in invalid_dims:
            with self.subTest(dims=dims):
                transport = FakeContractTransport(contract)
                transport.config["config"]["input"][0]["dims"] = dims
                with self.assertRaisesRegex(
                    TritonContractError,
                    "sbs_image.*config input IMAGE_BYTES dims.*signed decimal string",
                ):
                    await discover_model_capability(
                        transport, contract, target_batch=1
                    )

    async def test_remote_max_four_and_target_two_are_preserved(self):
        contract = MODEL_CONTRACTS["face_det"]
        transport = FakeContractTransport(contract, server_max=4)

        capability = await discover_model_capability(
            transport, contract, target_batch=2
        )

        self.assertEqual(capability.server_max_batch, 4)
        self.assertEqual(capability.target_batch, 2)

    async def test_target_above_server_max_fails_without_clamping(self):
        contract = MODEL_CONTRACTS["face_det"]
        transport = FakeContractTransport(contract, server_max=1)

        with self.assertRaises(TritonContractError) as caught:
            await discover_model_capability(transport, contract, target_batch=4)

        message = str(caught.exception)
        self.assertIn("face_det", message)
        self.assertIn("target 4", message)
        self.assertIn("server 1", message)

    async def test_config_type_string_matches_client_bytes(self):
        contract = MODEL_CONTRACTS["clip_image"]
        transport = FakeContractTransport(contract)

        capability = await discover_model_capability(
            transport, contract, target_batch=1
        )

        self.assertEqual(capability.key, "clip_image")

    async def test_capability_is_frozen_and_contains_remote_identity(self):
        contract = MODEL_CONTRACTS["face_det"]
        capability = await discover_model_capability(
            FakeContractTransport(contract), contract, target_batch=1
        )

        self.assertEqual(
            capability,
            ModelCapability(
                key="face_det",
                model_name="yolov8n_face_ensemble",
                api_contract_version=2,
                server_max_batch=1,
                target_batch=1,
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            capability.target_batch = 2

    async def test_model_must_be_ready_before_contract_is_fetched(self):
        contract = MODEL_CONTRACTS["face_det"]
        transport = FakeContractTransport(contract, ready=False)

        with self.assertRaisesRegex(TritonContractError, "face_det.*not ready"):
            await discover_model_capability(transport, contract, target_batch=1)

        self.assertEqual(transport.calls, [("ready", contract.model_name)])

    async def test_config_wrapper_must_contain_a_config_object(self):
        contract = MODEL_CONTRACTS["face_det"]
        for wrapper in ({}, {"config": None}, {"config": []}):
            with self.subTest(wrapper=wrapper):
                transport = FakeContractTransport(contract)
                transport.config = wrapper
                with self.assertRaisesRegex(
                    TritonContractError, "face_det.*config wrapper"
                ):
                    await discover_model_capability(
                        transport, contract, target_batch=1
                    )

    async def test_remote_model_names_must_match_contract(self):
        contract = MODEL_CONTRACTS["face_det"]
        for source in ("config", "metadata"):
            with self.subTest(source=source):
                transport = FakeContractTransport(contract)
                if source == "config":
                    transport.config["config"]["name"] = "other_model"
                else:
                    transport.metadata["name"] = "other_model"
                with self.assertRaisesRegex(
                    TritonContractError, f"face_det.*{source}.*model name"
                ):
                    await discover_model_capability(
                        transport, contract, target_batch=1
                    )

    async def test_server_max_batch_must_be_an_integer_at_least_one(self):
        contract = MODEL_CONTRACTS["face_det"]
        for value in (None, 0, -1, "not-an-int", 1.5, True):
            with self.subTest(value=value):
                transport = FakeContractTransport(contract)
                transport.config["config"]["max_batch_size"] = value
                with self.assertRaisesRegex(
                    TritonContractError, "face_det.*max_batch_size"
                ):
                    await discover_model_capability(
                        transport, contract, target_batch=1
                    )

    async def test_server_max_batch_rejects_a_numeric_string(self):
        contract = MODEL_CONTRACTS["face_det"]
        transport = FakeContractTransport(contract)
        transport.config["config"]["max_batch_size"] = "1"

        with self.assertRaisesRegex(
            TritonContractError, "face_det.*max_batch_size.*integer"
        ):
            await discover_model_capability(transport, contract, target_batch=1)

    async def test_contract_version_must_exist_be_positive_and_match(self):
        contract = MODEL_CONTRACTS["face_det"]
        cases = (
            ({}, "missing"),
            ({"api_contract_version": {}}, "string_value"),
            (
                {"api_contract_version": {"string_value": "x"}},
                "positive numeric string",
            ),
            (
                {"api_contract_version": {"string_value": "0"}},
                "positive numeric string",
            ),
            ({"api_contract_version": {"string_value": "1"}}, "expected 2"),
        )
        for parameters, expected_message in cases:
            with self.subTest(parameters=parameters):
                transport = FakeContractTransport(contract)
                transport.config["config"]["parameters"] = parameters
                with self.assertRaisesRegex(
                    TritonContractError,
                    f"face_det.*api_contract_version.*{expected_message}",
                ):
                    await discover_model_capability(
                        transport, contract, target_batch=1
                    )

    async def test_contract_version_requires_a_positive_numeric_string(self):
        contract = MODEL_CONTRACTS["face_det"]
        for value in (1, "", "1.0", "-1"):
            with self.subTest(value=value):
                transport = FakeContractTransport(contract)
                transport.config["config"]["parameters"] = {
                    "api_contract_version": {"string_value": value}
                }
                with self.assertRaisesRegex(
                    TritonContractError,
                    "face_det.*api_contract_version.*numeric string",
                ):
                    await discover_model_capability(
                        transport, contract, target_batch=1
                    )

    async def test_config_inputs_detect_missing_extra_duplicate_type_and_dims(self):
        contract = MODEL_CONTRACTS["face_rec"]
        base = model_config_json(contract, 1)
        mutations = {
            "missing": lambda items: items.pop(),
            "extra": lambda items: items.append(
                {"name": "EXTRA", "data_type": "TYPE_FP32", "dims": [1]}
            ),
            "duplicate": lambda items: items.append(deepcopy(items[0])),
            "datatype": lambda items: items[0].update(data_type="TYPE_FP32"),
            "dims": lambda items: items[0].update(dims=[1, 2, 3]),
        }
        for difference, mutate in mutations.items():
            with self.subTest(difference=difference):
                transport = FakeContractTransport(contract)
                transport.config = deepcopy(base)
                mutate(transport.config["config"]["input"])
                with self.assertRaisesRegex(
                    TritonContractError, f"face_rec.*config input.*{difference}"
                ):
                    await discover_model_capability(
                        transport, contract, target_batch=1
                    )

    async def test_config_outputs_detect_missing_extra_duplicate_type_and_dims(self):
        contract = MODEL_CONTRACTS["person_seg"]
        base = model_config_json(contract, 1)
        mutations = {
            "missing": lambda items: items.pop(),
            "extra": lambda items: items.append(
                {"name": "EXTRA", "data_type": "TYPE_FP32", "dims": [1]}
            ),
            "duplicate": lambda items: items.append(deepcopy(items[0])),
            "datatype": lambda items: items[0].update(data_type="TYPE_UINT8"),
            "dims": lambda items: items[0].update(dims=[1, 2, 3]),
        }
        for difference, mutate in mutations.items():
            with self.subTest(difference=difference):
                transport = FakeContractTransport(contract)
                transport.config = deepcopy(base)
                mutate(transport.config["config"]["output"])
                with self.assertRaisesRegex(
                    TritonContractError,
                    f"person_seg.*config output.*{difference}",
                ):
                    await discover_model_capability(
                        transport, contract, target_batch=1
                    )

    async def test_config_renamed_tensor_reports_missing_and_extra(self):
        contract = MODEL_CONTRACTS["face_det"]
        transport = FakeContractTransport(contract)
        transport.config["config"]["input"][0]["name"] = "RENAMED"

        with self.assertRaises(TritonContractError) as caught:
            await discover_model_capability(transport, contract, target_batch=1)

        message = str(caught.exception)
        self.assertIn("missing", message)
        self.assertIn("RAW_IMAGE", message)
        self.assertIn("extra", message)
        self.assertIn("RENAMED", message)

    async def test_metadata_detects_missing_extra_duplicate_and_type_differences(self):
        contract = MODEL_CONTRACTS["person_seg"]
        for section in ("inputs", "outputs"):
            mutations = {
                "missing": lambda items: items.pop(),
                "extra": lambda items: items.append(
                    {"name": "EXTRA", "datatype": "FP32"}
                ),
                "duplicate": lambda items: items.append(deepcopy(items[0])),
                "datatype": lambda items: items[0].update(datatype="INT64"),
            }
            for difference, mutate in mutations.items():
                with self.subTest(section=section, difference=difference):
                    transport = FakeContractTransport(contract)
                    mutate(transport.metadata[section])
                    with self.assertRaisesRegex(
                        TritonContractError,
                        f"person_seg.*metadata {section[:-1]}.*{difference}",
                    ):
                        await discover_model_capability(
                            transport, contract, target_batch=1
                        )

    async def test_target_batch_must_be_an_integer_at_least_one(self):
        contract = MODEL_CONTRACTS["face_det"]
        for target in (0, -1, 1.5, True):
            with self.subTest(target=target):
                with self.assertRaisesRegex(
                    TritonContractError, "face_det.*target_batch.*at least 1"
                ):
                    await discover_model_capability(
                        FakeContractTransport(contract),
                        contract,
                        target_batch=target,
                    )


class DiscoverAllCapabilitiesTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_models_are_discovered_into_keyed_mapping(self):
        contracts = {
            key: MODEL_CONTRACTS[key] for key in ("face_det", "person_pose")
        }
        transport = FakeMultiContractTransport(contracts)

        capabilities = await discover_all_capabilities(
            transport, contracts, {"face_det": 1, "person_pose": 1}
        )

        self.assertEqual(list(capabilities), ["face_det", "person_pose"])
        self.assertEqual(capabilities["face_det"].server_max_batch, 1)

    async def test_all_model_errors_are_aggregated_without_first_error_short_circuit(self):
        contracts = {
            key: MODEL_CONTRACTS[key] for key in ("face_det", "person_pose")
        }
        transport = FakeMultiContractTransport(contracts)
        transport.ready[contracts["face_det"].model_name] = False
        transport.ready[contracts["person_pose"].model_name] = False

        with self.assertRaises(TritonContractError) as caught:
            await discover_all_capabilities(
                transport, contracts, {"face_det": 1, "person_pose": 1}
            )

        message = str(caught.exception)
        self.assertIn("face_det", message)
        self.assertIn("person_pose", message)
        self.assertNotIn("face_det: face_det:", message)
        self.assertNotIn("person_pose: person_pose:", message)

    async def test_missing_target_is_reported_as_a_model_contract_error(self):
        contracts = {"face_det": MODEL_CONTRACTS["face_det"]}

        with self.assertRaisesRegex(
            TritonContractError, "face_det.*target_batch.*missing"
        ):
            await discover_all_capabilities(
                FakeMultiContractTransport(contracts), contracts, {}
            )

    async def test_cancelled_discovery_is_not_wrapped_as_contract_error(self):
        contracts = {"face_det": MODEL_CONTRACTS["face_det"]}
        transport = FakeMultiContractTransport(contracts)
        transport.ready[contracts["face_det"].model_name] = asyncio.CancelledError()

        with self.assertRaises(asyncio.CancelledError):
            await discover_all_capabilities(
                transport, contracts, {"face_det": 1}
            )


if __name__ == "__main__":
    unittest.main()
