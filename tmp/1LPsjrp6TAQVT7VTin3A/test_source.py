import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import source

def test_calculate_max_rolling_range():
    power_ac = pd.Series([10, 15, 20, 10, 25, 30, 35, 20, 25, 30])
    roll_periods = 3
    expected_result = 50.0
    result = source._calculate_max_rolling_range(power_ac, roll_periods)
    assert result == expected_result, f'Expected {expected_result} but got {result}'