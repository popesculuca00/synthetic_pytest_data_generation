def client_color_to_rgb(value: int):
    
    if value < 0 or value > 215:
        return 0
    return ((value // 36 * 0x33) << 16) + ((value // 6 % 6 * 0x33) << 8) + ((value % 6 * 0x33) & 0xFF)