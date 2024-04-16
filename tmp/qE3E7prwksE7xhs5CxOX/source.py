def convert_timestamp(timestamp: float, tenthousandths=False):
    
    from datetime import datetime
    if tenthousandths:
        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
    else:
        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return readable_time