from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib
from cycler import cycler
import pathlib
import seaborn as sns

import matplotlibheader as mplhead

import utils, plotter

def exponential_decay(dists, L=None, beta=1., alpha=.1):
    """ Somple visualisation of exponentially decaying dependancy for
    chance to connect as proposed by Waxman.

    Parameters
    ----------

    Returns
    -------

    """

    # dist = sp.spatial.distance.cdist(x, x)

    if L is None:
        L = max(dists)

    p = beta * np.exp(-dists / (alpha * L))

    return p


def lognorm_decay(dists, L=None, sigma=10, mu=0.00):
    """ Somple visualisation of exponentially decaying dependancy for
    chance to connect as proposed by Waxman.

    Parameters
    ----------

    Returns
    -------

    """
    if L is None:
        L = max(dists)

    dists = dists / L
    p = ( 1 / (np.sqrt(2 * np.pi) * sigma * dists)) * np.exp(
        (-1 / 2) * ((np.log(dists) - mu) / sigma)**2)

    return p


def powerlaw_decay(dists, L=None, beta=0.001, alpha=5/2):
    """ Somple visualisation of exponentially decaying dependancy for
    chance to connect as proposed by Waxman.

    Parameters
    ----------

    Returns
    -------

    """
    if L is None:
        L = max(dists)

    p = beta * (dists / L)**(-alpha)

    return p


def pareto_decay(dists, L=None, beta=1/2, alpha=.001):
    """ Somple visualisation of exponentially decaying dependancy for
    chance to connect as proposed by Waxman.

    Parameters
    ----------

    Returns
    -------

    """
    if L is None:
        L = max(dists)

    p = beta * alpha**beta * 1 / (dists/L)**(beta + 1)

    return p


def beta_decay(dists, L=None, b=5):
    """ Somple visualisation of exponentially decaying dependancy for
    chance to connect as proposed by Waxman.

    Parameters
    ----------

    Returns
    -------

    """

    beta_f = sp.special.beta(a, b)
    if L is None:
        L = np.max(dists)

    dists = dists / L

    p = 1 / beta_f * dists**(a-1) * (1 - dists)**(b-1)

    return p


if __name__ == "__main__":

    sns.set_theme(style="whitegrid")
    top_path = pathlib.Path.cwd()
    position_path = pathlib.Path.cwd() / 'results'

    res_path = top_path / 'results'
    res_path.mkdir(exist_ok=True)

    graph_path = top_path / 'graphs'
    graph_path.mkdir(parents=True, exist_ok=True)

    fdir = top_path / 'figures-10e-4'
    fdir.mkdir(exist_ok=True)

    figure_S1 = fdir / 'figure_S1'
    figure_S1.mkdir(exist_ok=True)

    # ext = '*pl*50*2_.csv'
    ext = '*2_.csv'
    paths = utils.get_files(res_path, ext)
    image_types = ['.png', '.svg', '.pdf']
    
    L = 1
    min_d = 1e-4
    # min_d = 10e-4
    x = np.linspace(min_d, L, 2001)
    
    fig, axs = plt.subplots(2, 1, sharex=False, figsize=(120*mplhead.mm, 140*mplhead.mm), constrained_layout=True)
    axs[0].set_prop_cycle(mplhead.cb_cycler_lines)
    axs[0].plot(x / L, exponential_decay(x, alpha=.059, beta=1), label='Exponential', linewidth=2)
    axs[0].plot(x / L, lognorm_decay(x, sigma=65.4, mu=0), label='Log-normal', linewidth=2)

    axs[1].set_prop_cycle(mplhead.cb_cycler_lines)
    axs[1].plot(x / L, exponential_decay(x, alpha=.059, beta=1),
                label='Exponential', linewidth=2)
    axs[1].plot(x / L, lognorm_decay(x, sigma=65.4, mu=0),
                label='Log-normal', linewidth=2)
    axs[0].vlines(2*min_d, 0, 2, linestyles='dotted',
                  label=r'Minimum node distance')
    # axs[0].vlines(2*10*min_d, 0, 2, linestyles='dotted',
    #               label=r'Minimum distance = $2 \cdot 10^{-3}$')
    axs[1].vlines(2*min_d, 0, 102, linestyles='dotted', label=r'Minimum node distance')
    # axs[1].vlines(2*10*min_d, 0, 102, linestyles='dotted',
    #               label=r'Minimum distance = $2 \cdot 10^{-3}$')

    print(lognorm_decay(4*min_d, sigma=65.4, mu=0, L=1))
    print(exponential_decay(4*min_d, alpha=.059, beta=1, L=1))
    axs[0].legend()
    axs[0].set_ylabel(r'$P(d)$')
    axs[0].set_xlabel(r'$d$')
    axs[1].set_ylabel(r'$P(d)$')
    axs[1].set_xlabel(r'$d$')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[0].set_ylim(0, 1)
    axs[0] = plotter.crisp_axis(axs[0])
    axs[1] = plotter.crisp_axis(axs[1])
    # axs[1].set_ylim(min_d, 2)

    fig_id = ['a', 'b', 'c']
    axs[0].set_title(fig_id[0], loc='left', fontsize=14, fontweight='bold')
    axs[1].set_title(fig_id[1], loc='left', fontsize=14, fontweight='bold')

    # fig.tight_layout(pad=0.12)

    plotter.savefigs(fig, figure_S1, 'figure_S1_P_d', itypes=image_types)

    # # fig, axs = plt.subplots(1, 2, figsize=(20, 9))

    # axs[0].set_prop_cycle(cb_cycler)
    # axs[0].plot(x / L, exponential_decay(x), label='Exponential decay', linewidth=3)
    # axs[0].plot(x / L,
    #         exponential_decay(x, beta=1., alpha=0.025),
    #         label='Exponential Onesto 2021',
    #         linewidth=3)
    # axs[0].plot(x / L,
    #         exponential_decay(x, beta=2.5, alpha=1/8),
    #         label='Exponential Macaque, Kaiser et al.',
    #         linewidth=3)
    # axs[0].plot(x / L,
    #         exponential_decay(x, beta=2.5, alpha=1/5),
    #         label='Exponential Cat, Kaiser et al.',
    #         linewidth=3)
    # sigma = 0.001

    # axs[0].plot(x / L, sigma*powerlaw_decay(x), label='Power-law decay', linewidth=3)
    # axs[0].plot(x / L, lognorm_decay(x, sigma=15), label='Log-normal decay', linewidth=3)
    # # axs[0].plot(x / L, lognorm_decay(x, sigma=20, mu=0.201), label='Log-normal macaque decay', linewidth=3)
    # # axs[0].plot(x / L, lognorm_decay(x, sigma=2, mu=0.001), label='Log-normal cat decay', linewidth=3)

    # # axs[0].plot(x / L, pareto_decay(x), label='Pareto decay', linewidth=3)
    # # axs[0].plot(x / L, beta_decay(x), label='Beta decay', linewidth=3)
    # axs[0].set_ylim(0.0001,2)
    # axs[0].set_xlabel('x/L')
    # axs[0].set_ylabel('P(x)')
    # axs[0].legend()

    # axs[1].set_prop_cycle(cb_cycler)
    # axs[1].plot(x / L,
    #             exponential_decay(x),
    #             label='Exponential decay',
    #             linewidth=3)
    # axs[1].plot(x / L,
    #             exponential_decay(x, beta=1., alpha=0.025),
    #             label='Exponential Onesto 2021',
    #             linewidth=3)
    # axs[1].plot(x / L,
    #             exponential_decay(x, beta=2.5, alpha=1 / 8),
    #             label='Exponential Macaque, Kaiser et al.',
    #             linewidth=3)
    # axs[1].plot(x / L,
    #             exponential_decay(x, beta=2.5, alpha=1 / 5),
    #             label='Exponential Cat, Kaiser et al.',
    #             linewidth=3)

    # sigma = 0.001

    # axs[1].plot(x / L,
    #             sigma * powerlaw_decay(x),
    #             label='Power-law decay',
    #             linewidth=3)
    # axs[1].plot(x / L, lognorm_decay(x), label='Log-normal decay', linewidth=3)
    # # axs[1].plot(x / L, pareto_decay(x), label='Pareto decay', linewidth=3)
    # # axs[1].plot(x / L, beta_decay(x), label='Beta decay', linewidth=3)
    # axs[1].set_xscale('log')
    # axs[1].set_yscale('log')
    # axs[1].set_xlabel('x/L')
    # axs[1].set_ylabel('P(x)')

    # fig.savefig('decay.png')
    # # fig.savefig('decay_log.png')


    # expo = exponential_decay(x)

    # fig, axs = plt.subplots(1, 2, figsize=(12, 9))
    # axs[0].set_prop_cycle(cb_cycler)
    # axs[1].set_prop_cycle(cb_cycler)

    # alphas = np.linspace(0.1, 1, 10)

    # for alpha in alphas:
    #     axs[0].plot(x / L, exponential_decay(x, alpha=alpha), label=('Exponential with alpha'+ str(np.round(alpha, 2))), linewidth=3)
    #     axs[1].plot(x / L,
    #                 exponential_decay(x, alpha=alpha),
    #                 label=('Exponential with alpha' + str(np.round(alpha, 2))),
    #                 linewidth=3)

    # # axs[0].set_ylim(bottom=.01)
    # axs[0].legend()
    # axs[1].set_ylim(bottom=.01)
    # axs[1].legend()
    # axs[1].set_xscale('log')
    # axs[1].set_yscale('log')
    # axs[1].set_xlabel('x/L')
    # axs[0].set_ylabel('P(x)')
    # axs[0].set_xlabel('x/L')


    # fig.savefig('expo_range.png')

    # fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    # axs[0, 0].set_prop_cycle(cb_cycler)
    # axs[1, 0].set_prop_cycle(cb_cycler)
    # axs[0, 1].set_prop_cycle(cb_cycler)
    # axs[1, 1].set_prop_cycle(cb_cycler)
    # alphas = np.linspace(1., 5, 5)
    # sigmas = np.linspace(0.001, 0.01, 2)

    # for alpha in alphas:
    #     axs[0, 0].plot(x / L,
    #                 sigmas[0]*powerlaw_decay(x, alpha=alpha),
    #                 label=('PL with alpha' + str(np.round(alpha, 2)), ' and sigma' + str(np.round(sigmas[0],3))),
    #                 linewidth=3)
    #     axs[0,1].plot(x / L,
    #                 sigmas[0] * powerlaw_decay(x, alpha=alpha),
    #                 label=('PL with alpha' + str(np.round(alpha, 2)), ' and sigma' + str(np.round(sigmas[0],3))),
    #                 linewidth=3)
    #     axs[1, 0].plot(x / L,
    #                    sigmas[1]*powerlaw_decay(x, alpha=alpha),
    #                    label=('PL with alpha' + str(np.round(alpha, 2)),
    #                           ' and sigma' + str(np.round(sigmas[1], 3))),
    #                    linewidth=3)
    #     axs[1, 1].plot(x / L,
    #                    sigmas[1] * powerlaw_decay(x, alpha=alpha),
    #                    label=('PL with alpha' + str(np.round(alpha, 2)),
    #                           ' and sigma' + str(np.round(sigmas[1], 3))),
    #                    linewidth=3)

    # # axs[0].set_ylim(bottom=.01)
    # axs[1, 0].legend()
    # axs[1, 1].set_ylim(bottom=.01)
    # axs[1, 1].legend()
    # axs[1, 1].set_xscale('log')
    # axs[1, 1].set_yscale('log')
    # axs[1, 1].set_xlabel('x/L')
    # axs[1, 0].set_ylabel('P(x)')
    # axs[1, 0].set_xlabel('x/L')
    # axs[1, 0].set_ylim([0, .5])
    # # axs[0].set_ylim(bottom=.01)
    # axs[0, 0].legend()
    # axs[0,1].set_ylim(bottom=.01)
    # axs[0,1].legend()
    # axs[0,1].set_xscale('log')
    # axs[0,1].set_yscale('log')
    # axs[0,1].set_xlabel('x/L')
    # axs[0, 0].set_ylabel('P(x)')
    # axs[0, 0].set_xlabel('x/L')
    # axs[0, 0].set_ylim([0,.5])

    # fig.savefig('pl_range.png')


    # fig, axs = plt.subplots(1, 2, figsize=(12, 9))
    # axs[0].set_prop_cycle(cb_cycler)
    # axs[1].set_prop_cycle(cb_cycler)

    # sigmas = np.linspace(3.0, 50, 11)
    # print(sigmas)
    # mus = [.0]#np.linspace(0.01, 0.1, 3)

    # for sigma, mu in product(sigmas, mus):
    #     axs[0].plot(x / L,
    #                 lognorm_decay(x, sigma=sigma, mu=mu),
    #                 label=('LN with sigma' + str(np.round(sigma, 3))),
    #                 linewidth=3)
    #     axs[1].plot(x / L,
    #                 lognorm_decay(x, sigma=sigma, mu=mu),
    #                 label=('LN with sigma' + str(np.round(sigma, 3))),
    #                 linewidth=3)

    # # axs[0].set_ylim(bottom=.01)
    # axs[0].legend()
    # axs[1].set_ylim(bottom=.01)
    # axs[1].legend()
    # axs[1].set_xscale('log')
    # axs[1].set_yscale('log')
    # axs[1].set_xlabel('x/L')
    # axs[0].set_ylabel('P(x)')
    # axs[0].set_xlabel('x/L')
    # axs[0].set_ylim([0, .5])

    # fig.savefig('lognorm_range.png')


