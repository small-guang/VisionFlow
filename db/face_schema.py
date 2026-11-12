"""Face collection schema and startup compatibility validation."""

from pymilvus import DataType


FACE_COLLECTION_NAME = "alogFace"
FACE_EMBEDDING_DIM = 512
FACE_MAX_CAPACITY = 1000


def build_face_schema(client, dim=FACE_EMBEDDING_DIM):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field("faceId", DataType.VARCHAR, max_length=36, is_partition_key=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("deviceId", DataType.VARCHAR, max_length=255)
    schema.add_field("trackId", DataType.INT64)
    schema.add_field("keyframeId", DataType.INT64)
    schema.add_field("faceRect", DataType.JSON)
    schema.add_field("faceKeyPoints", DataType.JSON)
    schema.add_field("faceCropPath", DataType.VARCHAR, max_length=255)
    schema.add_field("timestamp", DataType.INT64)
    schema.add_field("qualityScore", DataType.DOUBLE, description="face quality score at enrollment")
    schema.add_field("sampleId", DataType.VARCHAR, max_length=36, description="idempotent observation id")
    return schema


def validate_face_schema(description, dim=FACE_EMBEDDING_DIM):
    fields = {field["name"]: field for field in description.get("fields", [])}
    required_types = {
        "id": DataType.INT64, "faceId": DataType.VARCHAR,
        "embedding": DataType.FLOAT_VECTOR, "deviceId": DataType.VARCHAR,
        "trackId": DataType.INT64, "keyframeId": DataType.INT64,
        "faceRect": DataType.JSON, "faceKeyPoints": DataType.JSON,
        "faceCropPath": DataType.VARCHAR, "timestamp": DataType.INT64,
        "qualityScore": DataType.DOUBLE, "sampleId": DataType.VARCHAR,
    }
    compatible = (
        not description.get("auto_id", False)
        and not fields.get("id", {}).get("auto_id", False)
        and fields.get("id", {}).get("is_primary", False)
        and fields.get("faceId", {}).get("is_partition_key", False)
        and all(fields.get(name, {}).get("type") == dtype for name, dtype in required_types.items())
        and int(fields.get("embedding", {}).get("params", {}).get("dim", 0)) == dim
    )
    if not compatible:
        raise RuntimeError(
            "Incompatible face gallery schema. Expected manual INT64 ids, a faceId "
            "partition key, qualityScore and sampleId. Migrate the collection before starting ALog."
        )
