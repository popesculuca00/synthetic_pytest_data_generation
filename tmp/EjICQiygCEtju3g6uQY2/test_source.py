Python
import sys
sys.path.append(".")
import source  # assuming the function resides in source.py

def test_EVI():
    assert source.EVI(0,0,0) == 0, "Test Failed: expected 0 but got something else"

def test_EVI_positive():
    assert source.EVI(1,2,3) > 0, "Test Failed: expected a positive number but got something else"

def test_EVI_zero():
    assert source.EVI(1,1,1) == 0, "Test Failed: expected 0 but got something else"

def test_EVI_negative():
    assert source.EVI(2,3,1) < 0, "Test Failed: expected a negative number but got something else"