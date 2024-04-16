import pytest
from source import boost_nph

def test_boost_nph():
    data = {'binding_score': [1.1, 2.2, 3.3, 4.4]}
    assert boost_nph(data, 10) == 4.0