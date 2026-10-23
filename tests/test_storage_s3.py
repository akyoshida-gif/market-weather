from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from mw.core.config import GuardSettings
from mw.core.storage import GuardedStorage, S3Compatible


@pytest.fixture
def s3_storage():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="mw-test")
        yield GuardedStorage(S3Compatible("mw-test", client=client), GuardSettings())


def test_s3_roundtrip(s3_storage):
    s3_storage.put_bytes("silver/x", b"hello")
    assert s3_storage.get_bytes("silver/x") == b"hello"
    assert s3_storage.exists("silver/x")
    assert not s3_storage.exists("silver/y")
    assert s3_storage.list("silver/") == ["silver/x"]
    with pytest.raises(FileNotFoundError):
        s3_storage.get_bytes("nope")
