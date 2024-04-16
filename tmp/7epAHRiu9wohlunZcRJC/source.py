def final_leaf_dict(leaf, left_right):
    
    if leaf[5] is None or leaf[6] is None or leaf[7] is None or (
            left_right is None):
        print(leaf)
        raise Exception('No valid entries in final leaf.')
    return_dic = {'x_name': leaf[5], 'x_type': leaf[6],
                  'cut-off or set': leaf[7], 'left or right': left_right}
    return return_dic