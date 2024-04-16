
import pytest
from source import lightness_correlate

def test_lightness_correlate():
    assert lightness_correlate(50, 100, 25, 50) == 100
    assert lightness_correlate(100, 100, 100, 100) == 10000
    assert lightness_correlate(1, 1, 1, 1) == 1

