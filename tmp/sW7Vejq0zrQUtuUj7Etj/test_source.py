import pytest
import source

def test_can_absorb():
    with pytest.raises(AttributeError):
        left = source.MyClass()
    with pytest.raises(AttributeError):
        right = source.MyClass()
    with pytest.raises(UnboundLocalError):
        assert source.can_absorb(left, right)