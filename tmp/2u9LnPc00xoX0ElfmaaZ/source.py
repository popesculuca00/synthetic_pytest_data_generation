def drawCutoffVert(ax,x,cl="r",lb="",ls="--",lw=.5):
    
    oldLims = ax.get_ylim()
    ax.vlines(x,ymin=ax.get_ylim()[0],ymax=ax.get_ylim()[1],color='r',label=lb,linestyle=ls,linewidth=lw)
    ax.set_ylim(oldLims)
    return ax