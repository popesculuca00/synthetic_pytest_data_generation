def find_missing_timestamps(timestamps):
    
    if timestamps is None or len(timestamps) == 0:
        return None

    min_correct_timestamp = timestamps[0] if timestamps[0] == 0 else 0
    max_correct_timestamp = timestamps[-1]
    correct_time_values_in_ps = (
        set(range(min_correct_timestamp,
                  max_correct_timestamp,
                  100))
    )
    missing_timestamps = correct_time_values_in_ps.difference(timestamps)
    return sorted(missing_timestamps)