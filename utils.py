"""
Module containing usefull functions for diverse tasks not related to modelling.
Includes auxilary task and functions to ease storing and reading files for
further analysis.

List of functions
-----------------
store_file
"""


import csv
import pathlib
 
import time
import numpy as np
import pandas as pd
import stats
import networkx as nx
import os
import coordinate


def to_bool(value: str) -> bool:
    """
    Converts a string to a boolean, raising an error for invalid values.
    Handles 'true'/'false' case-insensitively and strips whitespace.
    """
    clean_value = value.strip().lower()
    if clean_value == 'true':
        return True
    elif clean_value == 'false':
        return False
    else:
        raise ValueError(f"Invalid boolean string: '{value}'")

def combinations_to_run(*args, seed=12345):
    """ Simple method to make an array of all possible combinations of given
    args. Usefull for consecutive runs or parallell simulations. Each element
    must contain the same type for the same parameter (no nested ragged lists 
    etc.).

    Parameters
    ----------
    *args : float or 1-d array
        Arguments to meshgrid.
    seed : seed
        Returns streams for runs at last element, such that each has an
        independent stream.
    Returns
    -------
    grid : ndarray
        Array containing all possible combinations of input variables.
    """
    grid = np.array(np.meshgrid(*args, indexing='ij'),
                    dtype=object).T.reshape(-1, len(args))
    streams = stats.rng_streams(seed, len(grid))
    grid = np.append(grid.T, [streams], axis=0).T
    grid = list(map(tuple, grid))

    return grid


def store_arr_row(fname, x):
    """ Function to 1d array to file. Stores as txt.

    Parameters
    ----------
    fname : string
        Path to append file to.
    x : 1d array
        Row values to append to file.
    Returns
    -------
    None : None
        Stores row values to file.
    """

    with open(fname, 'a') as f:
        if f.tell():
            f.write('\n')
        np.array(x,dtype=object).tofile(f, sep=',')

    return None

# def store_csv(x, names, savepath = None):
#     """ Function to store multiple values to a given path.

#     Parameters
#     ----------
#     x : ndarray
#     names : ndarray
#     savepath : strng

#     Returns
#     -------
#     Nothing. Stores all variables given in res to savepath.
#     """

#     if savepath is None:
#         savepath = pathlib.Path.cwd()

#     for key in df:
#         print(df[key])
#         with open('dump.txt', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(list)

# for key, value in df.iteritems():
#     print(key, value)
#     print()

#     pd.DataFrame(dict(a=a.tolist(), b=b)).T

#     savefile =
#     f.savefig(current_name + '_loglog.png')

# # with open("dump.bin", "wb") as f:
# #     pickle.dump(list, f)

# with open('dump.txt', 'a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(list)

# def init_savedir(f, spath):
#     """ Initates directory and path for saving results.



"""
        # if savepath == None:
    #     savepath  = Path.cwd() / 'results'

    # if not savepath.exists():
    #     savepath.mkdir()        # same effect as exist_ok = True


###########################
#    MEMBRANE POTENTIAL   #
###########################


# def update_membrane_pot(u, w, tau=1):
#     Update membrane potential, u, of neurons
#

#     du = (-u/tau + (1-x_i) sum_k^N w_ij*f(x_k))

#     u_new = u + du
# return u_new
#     % input
#     input_current  = dt*repmat(E_max-neurons.mp,1,n_neurons).*repmat(neurons.activity',n_neurons,1).*neurons.connectivity;

#     % update neuronal membrane potential
#     neurons.mp = neurons.mp-neurons.mp/exp(1)+nansum(input_current,2);

#     % keep membrane potential within the limits
#     neurons.mp(neurons.mp>E_max) = E_max;
#     neurons.mp(neurons.mp<E_min) = E_min;
"""

def get_rfiles(fpath, ext):
    """ Finds all files with given extension in all subdirectories.

    Parameters
    ----------
    fpath : posixpath
        Directory path to search files.
    ext : string
        String for search value.

    Returns
    -------
    files : list
        List containing all files in directory with given extension as
        popsixpath elements.
    """

    files = []

    for current_file in pathlib.Path(fpath).rglob(ext):
        files.append(current_file)

    return files


def get_files(fpath, ext):
    """ Finds all files with given extension in directory.

    Parameters
    ----------
    fpath : posixpath
        Directory path to search files.
    ext : string
        String for search value.

    Returns
    -------
    files : list
        List containing all files in directory with given extension as
        popsixpath elements.
    """

    files = []

    for current_file in pathlib.Path(fpath).glob(ext):
        files.append(current_file)

    return files


def to_dict(df, by='Mea'):
    """ Converts a DataFrame to dict grouped by given string

    Parameters
    ----------
    df : DataFrame
        DataFrame containing spiketimes from
    by : string
        String to group by (default value is 'Mea')

    Returns
    -------
    mea : dict
        Dict containing new DataFrames for each individual well.
    """

    mea = dict(tuple(df.groupby(by)))

    return mea


def to_bin(df, f, times='Time (s)', sender='Electrode'):
    """ Takes a dataframe containing spike times for senders, and returns a
    binary matrix for these as int.

    Parameters
    ----------
    df : DataFrame
        Spike times for senders.
    f : float
        Recording frequency.
    times : string
        Column name for spike times.
    sender : string
        Column name for senders of given spike.
    Returns
    -------
    bins : DataFrame
        DataFrame containing binary matrix for spikes. Labeled by electrode id.
    """

    bin_list = np.zeros((df[times].max() * f).astype(int) + 1)
    grouped = df.groupby(sender)
    bin_dict = {}

    for name, group in grouped:
        bin_list[(group[times] * f).astype(int)] = 1.0
        bin_dict[name] = bin_list
        bin_list = np.zeros((df[times].max() * f).astype(int) + 1)

    bins = pd.DataFrame(data=bin_dict)

    return bins


def shuffle_corr(times, mean, mu):
    """ Shuffles and ruffles time series data stored in

    """
    return None


def file_iter(f):
    """
    Checks if file exists. if not, returns file pluss one.

    Parameters
    ----------
    f

    Returns
    -------
    """
    n = 0

    while os.path.exists(img_nme):
        n += 1
        img_nme = 'bins' + str(n) + '.png'

    return f


def save_file(id, ext, path_var=None, sdir=None):
    """ Create and return savepath based on result folder structure.

    Parameters
    ----------

    Returns
    -------

    """

    if sdir == None:
        sdir = pathlib.Path.cwd() / 'results'
        sdir.mkdir(exist_ok=True)

    f = str()

    if path_var != None:
        for p in path_var:
            f = f + str(p) + '_'

    f = str(sdir / str(f + id + ext))

    return f


def sum_string(*args, divider='_'):
    """ Convert and append strings in givn input args with given divider. 
    Unpack parameters when sending them (* before list etc.).
    
    Parameters
    ----------

    Returns
    -------

    """
    long_string = str(args[0])

    for arg in args[1:-1]:
        long_string = long_string + '_' + str(arg)

    long_string = long_string + '_' + str(args[-1])
    
    return long_string


def cpr(df, f, col, rval):
    """ Function to copy, move and remove certain rows of the given Dataframe.

    Parameters
    ----------
    df:

    f:

    rm:

    Returns
    -------
    Not applicable
        Stores copy if file in given location.

    """

    df = drop_row(df, col, rval)

    return None


def drop_row(df, col, rval):
    """ Removes rows with value given.

    Parameters
    ----------
    df:

    rm:

    Returns
    -------
    df
    """

    # df.drop(df.loc[df[col].isin(rval) == True].index, inplace=True)
    # df = df[df[col].isin(rval) == False]
    df = df.loc[~df[col].isin(rval)]

    return df


def describe(G):
    """ Simple function to describe input graph G.
    """
    print(G)
    return G

def describe_minmax_distance(pos, r, periodic):
    """ Simple function to check min/max distances.
    """
    dists = coordinate.periodic_dist(pos, periodic=periodic)
    if periodic:
        print(f'Expected for periodic max is sqrt(1/2): {np.sqrt(1/2)}')
    else:
        print(f'Expected for non-periodic max is sqrt(2): {np.sqrt(2)}')
        
    print(f'Calculated max for pos with periodic set to {str(periodic)}: {np.max(dists)}')
    print(f'Calculated min for pos with periodic set to {str(periodic)}: {np.min(dists[np.where(dists!=.0)]):.5f}')
    print(f'In close proximity(<2*r*1.1 with r={r}): {len(dists[np.where(dists<2*r*1.1)])-len(dists)}')

    return None


def uni_run(params):
    box_len = len(params[4])

    if box_len == 2:
        uni_df = 0
        pos = 0
        n = params[2]
        r = params[3]
        box = params[4]
        periodic = params[5]
        version_string = params[6]
        seed = params[-1]

        f_name = sum_string(*params[0:4], len(params[4]))
        ext = '.csv'
        sdir = pathlib.Path.cwd() / f'results_periodic_{str(periodic)}_{version_string}'
        sdir.mkdir(exist_ok=True)
        sdir = sdir / str(params[1])
        sdir.mkdir(exist_ok=True)
        sdir = sdir / str(params[2])
        sdir.mkdir(exist_ok=True)
        sdir = sdir / str(params[3])
        sdir.mkdir(exist_ok=True)
        sfile = utils.save_file(f_name, ext, sdir=sdir)

        # if not pathlib.Path.exists(pathlib.Path(sfile)):
        pos = coordinate.uni_place(
            n=n, r=r, box=box, periodic=periodic, 
            seed=seed)
        
        describe_minmax_distance(pos, r=r, periodic=periodic)

        uni_df = coordinate.pos_init2d(len(pos), pos=pos)

        uni_df.to_csv(sfile)

        param_name = params[1] + '_params'
        res_dir = pathlib.Path.cwd() / f'results_n_{str(n)}_r_{str(r)}_{str(periodic)}_{version_string}'
        res_dir.mkdir(exist_ok=True)
        res_file = save_file(param_name, ext, sdir=res_dir)

        utils.store_arr_row(res_file, params)


def pl_run(params):
    box_len = len(params[5])
    
    if box_len == 2:
        pl_df = 0
        pos = 0
        n = params[2]
        r = params[3]
        a = params[4]
        box = params[5]
        home_frac = params[6]
        periodic = params[7]
        version_string = params[8]
        seed = params[-1]

        f_name = utils.sum_string(*params[0:4], str(int(params[6] * params[2])),
                                len(params[5]))
        ext = '.csv'
        sdir = pathlib.Path.cwd() / f'results_periodic_{str(periodic)}_{version_string}'
        sdir.mkdir(exist_ok=True)
        sdir = sdir / str(params[1])
        sdir.mkdir(exist_ok=True)
        sdir = sdir / str(params[2])      
        sdir.mkdir(exist_ok=True)
        sdir = sdir / str(params[3])
        sdir.mkdir(exist_ok=True)
        sfile = utils.save_file(f_name, ext, sdir=sdir)

        # if not pathlib.Path.exists(pathlib.Path(sfile)):
        pos, home = coordinate.sf_place(n=n,
                                        r=r,
                                        a=a,
                                        box=box,
                                        periodic=periodic,
                                        home=home_frac,
                                        seed=seed)
        
        describe_minmax_distance(pos, r=r, periodic=periodic)


        pl_df = coordinate.pos_init2d(len(pos), pos=pos, home=home)

        pl_df.to_csv(sfile)

        param_name = params[1] + '_params'
        res_dir = pathlib.Path.cwd() / f'results_n_{str(n)}_r_{str(r)}_{str(periodic)}_{version_string}'
        res_dir.mkdir(exist_ok=True)
        res_file = utils.save_file(param_name, ext, sdir=res_dir)

        utils.store_arr_row(res_file, params)


def timer(tic=None):
    """ Simple function to estimate start and stop time.
    """

    if tic is None:
        tic = time.time()           # Simple timer
        start_timeer = time.localtime()
        current_timeer = time.strftime("%H:%M:%S", start_timeer)
        print("Start time is: ", current_timeer)
        return tic

    elif isinstance(tic, float):
        end_timeer = time.localtime()
        current_timeer = time.strftime("%H:%M:%S", end_timeer)
        print("End time is: ", current_timeer)
        toc = time.time() - tic
        print("Total elapsed time is ", np.floor(toc / 60), "minutes and ", toc % 60,
            " seconds.")
        return toc