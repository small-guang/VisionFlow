import asyncio
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.infer.contracts import ModelContract, TensorContract
from core.infer.errors import TritonContractError


_SIGNED_DECIMAL_PATTERN = re.compile(r"(?:0|[1-9][0-9]*|-[1-9][0-9]*)\Z")
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


@dataclass(frozen=True)
class ModelCapability:
    key: str
    model_name: str
    api_contract_version: int
    server_max_batch: int
    target_batch: int


def _client_datatype(config_datatype: Any) -> Any:
    if config_datatype == "TYPE_STRING":
        return "BYTES"
    if isinstance(config_datatype, str) and config_datatype.startswith("TYPE_"):
        return config_datatype.removeprefix("TYPE_")
    return config_datatype


def _parse_positive_int_value(
    value: Any, label: str, differences: list[str]
) -> int | None:
    if type(value) is not int or value < 1:
        differences.append(f"{label} must be a positive integer, got {value!r}")
        return None
    return value


def _parse_positive_numeric_string(
    value: Any, label: str, differences: list[str]
) -> int | None:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdigit()
        or int(value) < 1
    ):
        differences.append(
            f"{label} must be a positive numeric string, got {value!r}"
        )
        return None
    return int(value)


def _parse_config_dims(
    value: Any, label: str, differences: list[str]
) -> tuple[int, ...] | None:
    if not isinstance(value, list):
        differences.append(
            f"{label} dims must be a list of signed decimal strings, "
            f"got {value!r}"
        )
        return None

    parsed: list[int] = []
    invalid = False
    for index, dimension in enumerate(value):
        if type(dimension) is not str or not _SIGNED_DECIMAL_PATTERN.fullmatch(
            dimension
        ):
            differences.append(
                f"{label} dims[{index}] must be a canonical signed decimal "
                f"string, got {dimension!r}"
            )
            invalid = True
            continue
        numeric_dimension = int(dimension)
        if not _INT64_MIN <= numeric_dimension <= _INT64_MAX:
            differences.append(
                f"{label} dims[{index}] signed decimal string is outside int64: "
                f"{dimension!r}"
            )
            invalid = True
            continue
        parsed.append(numeric_dimension)
    return None if invalid else tuple(parsed)


def _tensor_differences(
    label: str,
    remote: Any,
    expected: Sequence[TensorContract],
    *,
    config_format: bool,
) -> list[str]:
    if not isinstance(remote, list):
        return [f"{label} list is missing or invalid"]

    differences: list[str] = []
    by_name: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for index, item in enumerate(remote):
        if not isinstance(item, Mapping):
            differences.append(f"{label} item {index} is invalid")
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            differences.append(f"{label} item {index} name is invalid")
            continue
        if name in by_name:
            duplicates.add(name)
        else:
            by_name[name] = item

    if duplicates:
        differences.append(f"{label} duplicate: {', '.join(sorted(duplicates))}")

    expected_by_name = {tensor.name: tensor for tensor in expected}
    missing = set(expected_by_name) - set(by_name)
    extra = set(by_name) - set(expected_by_name)
    if missing:
        differences.append(f"{label} missing: {', '.join(sorted(missing))}")
    if extra:
        differences.append(f"{label} extra: {', '.join(sorted(extra))}")

    for name in sorted(set(expected_by_name) & set(by_name)):
        tensor = expected_by_name[name]
        item = by_name[name]
        datatype_field = "data_type" if config_format else "datatype"
        actual_datatype = item.get(datatype_field)
        if config_format:
            actual_datatype = _client_datatype(actual_datatype)
        if actual_datatype != tensor.datatype:
            differences.append(
                f"{label} {name} datatype mismatch: "
                f"expected {tensor.datatype}, got {actual_datatype!r}"
            )
        if config_format:
            raw_dims = item.get("dims")
            actual_dims = _parse_config_dims(raw_dims, f"{label} {name}", differences)
            if actual_dims is not None and actual_dims != tensor.dims:
                differences.append(
                    f"{label} {name} dims mismatch: "
                    f"expected {tensor.dims}, got {actual_dims!r}"
                )
    return differences


def _contract_error(
    key: str, differences: Sequence[str]
) -> TritonContractError:
    return TritonContractError(f"{key}: " + "; ".join(differences))


async def discover_model_capability(
    transport: Any,
    expected: ModelContract,
    *,
    target_batch: int,
) -> ModelCapability:
    if (
        isinstance(target_batch, bool)
        or not isinstance(target_batch, int)
        or target_batch < 1
    ):
        raise _contract_error(
            expected.key,
            [f"target_batch must be an integer at least 1, got {target_batch!r}"],
        )

    if not await transport.is_model_ready(expected.model_name):
        raise _contract_error(
            expected.key, [f"model {expected.model_name} is not ready"]
        )

    config_wrapper = await transport.get_model_config(expected.model_name)
    metadata = await transport.get_model_metadata(expected.model_name)
    if not isinstance(config_wrapper, Mapping) or not isinstance(
        config_wrapper.get("config"), Mapping
    ):
        raise _contract_error(
            expected.key, ["config wrapper must contain a config object"]
        )

    config = config_wrapper["config"]
    differences: list[str] = []
    if config.get("name") != expected.model_name:
        differences.append(
            "config model name mismatch: "
            f"expected {expected.model_name}, got {config.get('name')!r}"
        )

    server_max_batch = _parse_positive_int_value(
        config.get("max_batch_size"), "max_batch_size", differences
    )

    parameters = config.get("parameters")
    version_parameter = (
        parameters.get("api_contract_version")
        if isinstance(parameters, Mapping)
        else None
    )
    if version_parameter is None:
        differences.append("api_contract_version is missing")
        version = None
    elif not isinstance(version_parameter, Mapping) or (
        "string_value" not in version_parameter
    ):
        differences.append("api_contract_version string_value is missing")
        version = None
    else:
        version = _parse_positive_numeric_string(
            version_parameter.get("string_value"),
            "api_contract_version string_value",
            differences,
        )
        if version is not None and version != expected.api_contract_version:
            differences.append(
                "api_contract_version mismatch: "
                f"expected {expected.api_contract_version}, got {version}"
            )

    differences.extend(
        _tensor_differences(
            "config input",
            config.get("input"),
            expected.inputs,
            config_format=True,
        )
    )
    differences.extend(
        _tensor_differences(
            "config output",
            config.get("output"),
            expected.outputs,
            config_format=True,
        )
    )

    if not isinstance(metadata, Mapping):
        differences.append("metadata object is missing or invalid")
    else:
        if metadata.get("name") != expected.model_name:
            differences.append(
                "metadata model name mismatch: "
                f"expected {expected.model_name}, got {metadata.get('name')!r}"
            )
        differences.extend(
            _tensor_differences(
                "metadata input",
                metadata.get("inputs"),
                expected.inputs,
                config_format=False,
            )
        )
        differences.extend(
            _tensor_differences(
                "metadata output",
                metadata.get("outputs"),
                expected.outputs,
                config_format=False,
            )
        )

    if server_max_batch is not None and target_batch > server_max_batch:
        differences.append(
            f"batch target {target_batch} exceeds server maximum; "
            f"target {target_batch}, server {server_max_batch}"
        )

    if differences:
        raise _contract_error(expected.key, differences)

    return ModelCapability(
        key=expected.key,
        model_name=expected.model_name,
        api_contract_version=version,
        server_max_batch=server_max_batch,
        target_batch=target_batch,
    )


async def discover_all_capabilities(
    transport: Any,
    contracts: Mapping[str, ModelContract],
    targets: Mapping[str, int],
) -> dict[str, ModelCapability]:
    async def discover(
        key: str, contract: ModelContract
    ) -> ModelCapability:
        if key not in targets:
            raise _contract_error(key, ["target_batch is missing"])
        return await discover_model_capability(
            transport, contract, target_batch=targets[key]
        )

    keys = list(contracts)
    results = await asyncio.gather(
        *(discover(key, contracts[key]) for key in keys),
        return_exceptions=True,
    )

    failures: list[str] = []
    capabilities: dict[str, ModelCapability] = {}
    for key, result in zip(keys, results):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            message = str(result)
            prefix = f"{key}: "
            if message.startswith(prefix):
                message = message.removeprefix(prefix)
            failures.append(f"{key}: {message}")
        else:
            capabilities[key] = result
    if failures:
        raise TritonContractError(
            "Triton capability discovery failed: " + " | ".join(failures)
        )
    return capabilities
