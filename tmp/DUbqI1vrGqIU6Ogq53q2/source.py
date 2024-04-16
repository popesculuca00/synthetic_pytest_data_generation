def local_coord_to_global(in_coord, center_coord, max_x, max_y):
    
    
    new_coord_0 = center_coord[0]  + in_coord[0]-1
    new_coord_1 = center_coord[1]  + in_coord[1]-1
    
    # only return valid coordinates, do nothing if coordinates would be negative
    if new_coord_0 >= 0 and new_coord_1 >= 0 and new_coord_0 <= max_x and new_coord_1 <= max_y:
        return (new_coord_0, new_coord_1)