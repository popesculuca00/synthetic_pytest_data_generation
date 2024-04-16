def filter_df(freq_df, prediction=None, top=None):
    

    if prediction:
        freq_df = freq_df.loc[:, prediction]
    if top:
        freq_df = freq_df[freq_df.sum().nlargest(top).index]
    return freq_df