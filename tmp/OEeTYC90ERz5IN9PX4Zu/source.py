def plot_hist_basic(df, col):
    
    data = df[col]
    ax = data.hist(bins=20, normed=1, edgecolor='none', figsize=(10, 7), alpha=.5)
    ax.set_ylabel('Probability Density')
    ax.set_title(col)

    return ax