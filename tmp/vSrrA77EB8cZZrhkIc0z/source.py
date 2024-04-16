def _calculate_max_rolling_range(power_ac, roll_periods):
    
    # Calculate the maximum value over a forward-rolling window
    max_roll = power_ac.iloc[::-1].rolling(roll_periods).max()
    max_roll = max_roll.reindex(power_ac.index)
    # Calculate the minimum value over a forward-rolling window
    min_roll = power_ac.iloc[::-1].rolling(roll_periods).min()
    min_roll = min_roll.reindex(power_ac.index)
    # Calculate the maximum rolling range within the foward-rolling window
    rolling_range_max = (max_roll - min_roll)/((max_roll + min_roll)/2)*100
    return rolling_range_max