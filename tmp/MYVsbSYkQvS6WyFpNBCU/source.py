def std_dev(time_series, window_size=20, fwd_fill_to_end=0):
    
    if fwd_fill_to_end <= 0:
        std = time_series.rolling(window=window_size).std()
    else:
        std = time_series.rolling(window=window_size).std()
        std[-fwd_fill_to_end:] = std.iloc[-fwd_fill_to_end]

    
    std.fillna(method='backfill', inplace=True)
    return std