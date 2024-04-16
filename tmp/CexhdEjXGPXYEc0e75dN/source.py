def _plot_tree(ax, y, ntiles, show_quartiles):
    
    if show_quartiles:
        # Plot median
        ax.plot(ntiles[2], y, 'bo', markersize=4)
        # Plot quartile interval
        ax.errorbar(x=(ntiles[1], ntiles[3]), y=(y, y), linewidth=2, color='b')

    else:
        # Plot median
        ax.plot(ntiles[1], y, 'bo', markersize=4)

    # Plot outer interval
    ax.errorbar(x=(ntiles[0], ntiles[-1]), y=(y, y), linewidth=1, color='b')
    return ax