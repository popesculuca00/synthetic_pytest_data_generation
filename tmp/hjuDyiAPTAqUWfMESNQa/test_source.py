import sys
sys.path.append(".")
from source import valid_color

def test_valid_color_within_range():
    assert valid_color(5, 1, 10) == True

def test_valid_color_out_of_range():
    assert valid_color(20, 1, 10) == False
    
def test_valid_color_equal_min():
    assert valid_color(1, 1, 10) == True
    
def test_valid_color_equal_max():
    assert valid_color(10, 1, 10) == True