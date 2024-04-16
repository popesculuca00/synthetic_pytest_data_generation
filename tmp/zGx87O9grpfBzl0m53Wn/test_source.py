import sys
sys.path.append(".")  # Adds the current directory to the python path
from source import get_intervals  # Import the function from source.py

def test_get_intervals_length_three():
    timestamps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert get_intervals(timestamps, length=3) == [(2.5, 5.5), (5.5, 8.5)]

def test_get_intervals_length_two():
    timestamps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert get_intervals(timestamps, length=2) == [(2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]

def test_get_intervals_zero_length():
    timestamps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert get_intervals(timestamps, length=0) == [(4.5, 4.5)]

def test_get_intervals_negative_length():
    timestamps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert get_intervals(timestamps, length=-1) == []

def test_get_intervals_single_value():
    timestamps = [1]
    assert get_intervals(timestamps, length=3) == []