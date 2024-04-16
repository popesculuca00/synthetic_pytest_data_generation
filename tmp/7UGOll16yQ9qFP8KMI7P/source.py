def __rred(r_1, r_2):
    
    if (r_1 == 0 or abs(r_1) == float('inf')) and r_2 != 0:
        r_red = r_2
    elif (r_2 == 0 or abs(r_2) == float('inf')) and r_1 != 0:
        r_red = r_1
    elif (r_1 == 0 or abs(r_1) == float('inf')) and \
            (r_2 == 0 or abs(r_2) == float('inf')):
        r_red = 0
    elif r_1 == -r_2:
        r_red = 0
    else:
        r_red = 1 / (1 / r_1 + 1 / r_2)
    return r_red