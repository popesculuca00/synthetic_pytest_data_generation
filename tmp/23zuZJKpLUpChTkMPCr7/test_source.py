# test_source.py
import pytest
import source  # Assuming the source code is in a file named 'source.py' in the same directory

def test_func_positive_power():
    # Test if function returns correct output when input is positive
    assert source.func(2, 2, 3) == 8

def test_func_zero_power():
    # Test if function returns correct output when power is zero
    assert source.func(0, 2, 3) == 1

def test_func_negative_power():
    # Test if function returns correct output when power is negative
    assert source.func(1, 2, -3) == 0.125

def test_func_large_power():
    # Test if function returns correct output when power is large
    assert source.func(3, 2, 10**18) == 2**10**18