import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats as spstats
from sklearn.neighbors import KernelDensity

import coordinate


def density_plot(ax, p, n, domain=(0, 0, 1, 1), dkwargs=None, pkwargs=None):
    """ plots density of points for given input positions, p.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    for reference.

    """
    xmin, ymin, xmax, ymax = np.array_split(domain, 4)
    # print(xmin, ymax)
    # xmin = p[:,0].min()
    # xmax = p[:,0].max()
    # ymin = p[:,1].min()
    # ymax = p[:,1].max()
    # print(xmin, ymax)

    X, Y = np.mgrid[xmin[0]:xmax[0]:500j, ymin[0]:ymax[0]:500j]

    positions = np.vstack([X.ravel(), Y.ravel()])

    kernel = spstats.gaussian_kde(p.T, bw_method=.1)

    Z = np.reshape(kernel(positions).T, X.shape)

    ax.imshow(np.rot90(Z), cmap='coolwarm', extent=[xmin[0], xmax[0], ymin[0], ymax[0]])
    #cmap=plt.cm.gist_earth_r
    # ax.scatter(p[:n, 0], p[:n, 1], color='#ff7f0e',
    #            alpha=.7)
    # ax.scatter(p[n:, 0], p[n:, 1], color='#1f77b4',
    #            alpha=.3)

def savefigs(fig, fpath, fname, itypes):
    """ Just a simple function to store images as multiple types for papers and
    stuff ya know.

    Parameters
    ----------
    fig
    path
    fname
    itypes
    
    Returns
    -------
    Not applicable.
    """

    for itype in itypes:
        fig_path = fpath / (fname + itype)
        fig.savefig(fig_path, bbox_inches='tight')#, pad_inches=0)

    return fig


def neuron_distribution(ax, pos, box, periodic=True, **kwargs):
    """ Plot neuron to neuron distance. Add implementation for x, y , x?
    keep as axis level, add seperate function for xyz distributions
    
    """
    if periodic:
        dists = coordinate.periodic_dist(pos, box, periodic=periodic)
    else:
        dists = sp.spatial.distance.cdist(pos, pos)

    flat_dists = dists[np.nonzero(np.triu(dists))]

    return ax.hist(flat_dists, align='mid', **kwargs)


def neuron_n_nearest_neighbours(ax, pos, box, n, periodic=True, **kwargs):
    """ Plot neuron to neuron distance. Add implementation for x, y , x?
    keep as axis level, add seperate function for xyz distributions

    """

    if periodic:
        dists = coordinate.periodic_dist(pos, box)
    else:
        dists = sp.spatial.distance.cdist(pos, pos)

    sort_dists = np.sort(dists, axis=1)

    n_neighbours = sort_dists[:, 1:(n + 1)]

    return ax.hist(n_neighbours.flatten(), align='mid', **kwargs)


def crisp_axis(ax, axis='both', left=True, right=True, bottom=True, top=True,
               direction='in', blackbox=False, **kwargs):
    """
    Matplotlib keeps messing up plots. Reset axis and stuff here.
    """

    ax.tick_params(axis='both', left=left, right=right,
                   top=top, bottom=bottom,
                   direction=direction, which='both', **kwargs)

    ax.grid(False)
    if blackbox:
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
    else:
        for spine in ax.spines.values():
            ax.tick_params(axis='both', which='both', color=spine.get_edgecolor())
    return ax
