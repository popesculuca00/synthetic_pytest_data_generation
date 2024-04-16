def _plot_tree(ax, y, ntiles, show_quartiles, c, plot_kwargs):
    
    if show_quartiles:
        # Plot median
        ax.plot(ntiles[2], y, color=c,
                marker=plot_kwargs.get('marker', 'o'),
                markersize=plot_kwargs.get('markersize', 4))
        # Plot quartile interval
        ax.errorbar(x=(ntiles[1], ntiles[3]), y=(y, y),
                    linewidth=plot_kwargs.get('linewidth', 2),
                    color=c)

    else:
        # Plot median
        ax.plot(ntiles[1], y, marker=plot_kwargs.get('marker', 'o'),
                color=c, markersize=plot_kwargs.get('markersize', 4))

    # Plot outer interval
    ax.errorbar(x=(ntiles[0], ntiles[-1]), y=(y, y),
                linewidth=int(plot_kwargs.get('linewidth', 2)/2),
                color=c)

    return ax