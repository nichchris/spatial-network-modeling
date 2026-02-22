import matplotlib as mpl
import matplotlib.pyplot as plt

from cycler import cycler


#plt.style.use('~/.config/mpl/paper')
#['~/.config/mpl/paper'
#rc('text', usetex=True)
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
#mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams['figure.figsize'] = (9, 6)
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['text.usetex'] = False
mpl.rcParams.update({"axes.grid" : False})

# mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams['mathtext.rm'] = 'serif'

# CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf']#, '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00']
color_cycle = ["#e69f00",
               "#56b4e9",
               "#009e73",
               "#0072b2",
               "#d55e00",
               "#cc79a7",
               "#f0e442"
               # '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000' # WONG color palette
               # '#e41a1c', '#ff7f00', '#8272fa', '#9f7ea4', '#f46f78', '#b2b914', '#853723', '#4155a4', '#528e22',
               #  '#75cad9', '#e41a1c', '#dede00'
               ]
color_cycle_rev2 = [
               "#56b4e9",
               "#e69f00",
               "#009e73",
               "#0072b2",
               "#d55e00",
               "#cc79a7",
               "#f0e442"]
# color_cycle = [
#     '#4155a4', '#853723', '#528e22', '#8272fa', '#9f7ea4', '#655d66',
#     '#f46f78', '#b2b914', '#75cad9', '#adf296', '#377eb8', '#ff7f00',
#     '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'
# ]
linestyle_str = [
    ('solid', 'solid'),      # Same as (0, ()) or '-'
    ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    ('dashed', 'dashed'),    # Same as '--'
    ('dashdot', 'dashdot')]  # Same as '-.'
linestyle_tuple = [
    ('loosely dotted',        (0, (1, 10))),
    ('dotted',                (0, (1, 1))),
    ('densely dotted',        (0, (1, 1))),
    ('long dash with offset', (5, (10, 3))),
    ('loosely dashed',        (0, (5, 10))),
    ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),

    ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),

    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

linestyles = ['-', '--', '-.', ':', '-',
              '--', '-.', ':', '-', '--', '-.', ':']
linestyles_short = ['solid', 'dashed', 'dotted', 'dashdot',
                    (0, (5, 10)), (0, (3, 5, 1, 5)), (0, (1, 10))]

#custom_cycler = (cycler('color',CB_color_cycle) *

cb_cycler = (cycler(linestyle=linestyles) * cycler(color=color_cycle))
cb_cycler_lines = (cycler(linestyle=linestyles_short) +
             cycler(color=color_cycle))
cb_cycler_color = cycler(color=color_cycle)

cb_cycler_lines = (cycler(linestyle=linestyles_short) +
                   cycler(color=color_cycle))

# plt.rc('axes', prop_cycle=default_cycler)

# Figure size in mm
cm = 1/2.54
mm = cm/10.0
