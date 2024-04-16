import pytest
from source import convert_timestamp

def test_convert_timestamp():
    assert convert_timestamp(1640149152) == '2021-12-22 06:59:12'

def test_convert_timestamp_tenthousandths():
    assert convert_timestamp(1640149152.123, tenthousandths=True
    ) == '2021-12-22 06:59:12.123000'

def test_convert_timestamp_negative():
    with pytest.raises(OSError):
        assert convert_timestamp(-1) == '1969-12-31 23:59:59'