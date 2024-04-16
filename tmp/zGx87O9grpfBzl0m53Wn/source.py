def get_intervals(timestamps, length=3.0):
    

    intervals = []
    itr = 0
    ref_itr = 0

    while itr < len(timestamps):

        startTimeStamp = timestamps[ref_itr]

        while itr < len(timestamps) and timestamps[itr] <= startTimeStamp + length:
            itr += 1

        endTimeStamp = timestamps[itr - 1]
        midTimeStamp = (startTimeStamp + endTimeStamp) / 2

        # Check for zeros
        if midTimeStamp - length / 2 < 0:
            intervals.append((0, length))
        else:
            intervals.append((midTimeStamp - length / 2, midTimeStamp + length / 2))

        ref_itr = itr

    return intervals