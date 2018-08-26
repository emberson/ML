import numpy as np
import matplotlib
from matplotlib import cm as cm
matplotlib.use("PDF")
import pylab as py
import sklearn.feature_selection
from PlayoffData import PlayoffData
import os

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Input directory containing prepared data for each season
input_root_dir = "./data/prepared/"
input_template = "seriesYEAR_REPLACE.dat"

# NHL seasons to aggregate data from 
seasons = np.arange(2008, 2018+1)

# Training split fraction and seed
train_fraction = 0.7
seed_split     = 92

# Various plots
plot_dir = "plots/DataExploration/"
plot_correlation_mat = plot_dir + "correlation_matrix.pdf"
plot_feature_hist    = plot_dir + "hist-_FEATURE_.pdf"

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

def PlotHistogram(pf, x, y, f):
    """
    Plot histogram of feature f for total as well as class distributions.
    """

    # Format plot name
    pf = pf.replace("_FEATURE_", str(f))

    # Plot window
    left   = 0.11
    right  = 0.97
    bottom = 0.11
    top    = 0.97

    #
    # Prepare the data
    #

    x0 = x.iloc[np.where(y == 0)]
    x1 = x.iloc[np.where(y == 1)]
    x0_mean = x0.mean()
    x0_std  = x0.std()
    x1_mean = x1.mean()
    x1_std  = x1.std()

    #
    # Make the plot
    #

    fig_width_pt  = 400.
    inches_per_pt = 1. / 72.27
    fig_width     = fig_width_pt * inches_per_pt
    fig_height    = fig_width * 0.85
    fig_size      = [fig_width, fig_height]
    params        = {'backend': 'ps', 'axes.labelsize': 13, 'font.size': 12, \
                    'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12, \
                    'text.usetex': True, 'figure.figsize': fig_size}

    py.rcParams.update(params)
    fig = py.figure(1)
    py.clf()
    py.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=0., wspace=0.)

    ax = py.subplot(111)

    # Plot histograms
    py.hist(x, color="k", histtype="step", fill=False, lw=2, label=r"${\rm All}$")
    py.hist(x0, color="r", histtype="step", fill=False, lw=2, label=r"$Y = 0$")
    py.hist(x1, color="g", histtype="step", fill=False, lw=2, label=r"$Y = 1$")

    # Show mean and standard deviation of the two classes
    ymin, ymax = ax.get_ylim()
    py.vlines(x0_mean, ymin, ymax, color="r", linestyles="dotted")
    py.vlines(x1_mean, ymin, ymax, color="g", linestyles="dotted")
    py.axvspan(x0_mean-x0_std, x0_mean+x0_std, alpha=0.1, color="r")
    py.axvspan(x1_mean-x1_std, x1_mean+x1_std, alpha=0.1, color="g")

    lg = py.legend(loc="best", fancybox=True)
    lg.draw_frame(False)

    py.xlabel(LabelFMT([feature]))
    py.ylabel(r"${\rm Count}$")

    ax.set_ylim((ymin, ymax))

    py.savefig(pf)
    py.close()

    print("Histogram saved to " + pf)

def LabelFMT(labs):
    """
    Formats labs to be compatible with latex.
    """

    labs = np.array(labs)
    for i in range(labs.shape[0]): labs[i] = labs[i].replace("_", "-")

    return labs

def PlotCorrelationMatrix(pf, cmat):
    """
    Plots correlation matrix of the feature set.
    """

    #
    # Plot parameters
    #

    # Labels
    labs = LabelFMT(cmat.columns)

    # Colormap parameters
    cmap = cm.get_cmap("jet") 
    cmin = -1.
    cmax = 1.
    cspace = -0.03
    cwidth = 0.04

    # Plot window
    left   = 0.14
    right  = 0.86
    dw     = 1.0-(right-left)
    bottom = dw/2
    top    = 1 - dw/2

    #
    # Make the plot
    #

    fig_width_pt  = 400.
    inches_per_pt = 1. / 72.27
    fig_width     = fig_width_pt * inches_per_pt
    fig_height    = fig_width * 0.85
    fig_size      = [fig_width, fig_height]
    params        = {'backend': 'ps', 'axes.labelsize': 13, 'font.size': 12, \
                    'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12, \
                    'text.usetex': True, 'figure.figsize': fig_size}

    py.rcParams.update(params)
    fig = py.figure(1)
    py.clf()
    py.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=0., wspace=0.)

    ax1 = py.subplot(111)

    py.imshow(cmat, cmap=cmap, vmin=cmin, vmax=cmax, interpolation="nearest")

    bbox  = np.array(py.gca().get_position())
    cax  = fig.add_axes([bbox[1][0]+cspace, bbox[0][1], cwidth, bbox[1][1]-bbox[0][1]]) # (left xpos, bottom ypos, width, height) 
    cbar = py.colorbar(cax=cax)
    cbar.solids.set_edgecolor("face")

    ax1.set_xticks(np.arange(cmat.shape[0])) ; ax1.set_xticklabels(labs, rotation="vertical")
    ax1.set_yticks(np.arange(cmat.shape[0])) ; ax1.set_yticklabels(labs)#, rotation="vertical")

    py.savefig(pf)
    py.close()

    print("Correlation matrix plot saved to " + pf)

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

# Make output plot directory if it does not yet exist
if not os.path.isdir(plot_dir): os.makedirs(plot_dir)

#
# Gather playoff data over all seasons
#

data = PlayoffData(input_root_dir, input_template, seasons)

#
# Perform ANOVA test to rank features
#

data.SetSplitSeed(seed_split)
data.SplitTrainingTestingData(train_fraction)
select = sklearn.feature_selection.SelectKBest(k=data.num_features)
select_fit = select.fit(data.x_train, data.y_train)
scores  = select_fit.scores_
pvalues = select_fit.pvalues_
isort   = np.argsort(scores)[::-1]
cols    = data.features.columns[isort]
scores  = scores[isort]
pvalues = pvalues[isort] 

sl=20
print(3*sl*"-")
print("ANOVA Test".upper().center(3*sl))
print(3*sl*"-")
print("Feature".ljust(sl) + "F-Statistic".ljust(sl) + "p-value".ljust(sl))
for i in range(scores.shape[0]):
    print(str(cols[i]).ljust(sl) + str(scores[i]).ljust(sl) + str(pvalues[i]).ljust(sl))
print(3*sl*"-")

#
# Plot correlation matrix of the feature set
#

PlotCorrelationMatrix(plot_correlation_mat, data.features.corr())

#
# Plot histograms of each feature
#

for feature in data.features: PlotHistogram(plot_feature_hist, data.features[feature], data.target, feature) 

#
# From correlation matrix we find features that are within 0.75 correlation which are:
# 1) Home, PP, BB
# 2) CD, FD, etc.
# 3) GD, PDO, etc.
# We use ANOVA to rank which of these are most important in each group and find this to be:
# 1) BB
# 2) FD
# 3) GD
# So we would delete the other columns as in: 
#data.DropFeatures(["Home", "PP", "CD", "CD_N", "CD_M", "PDO", "PDO_N", "PDO_M", "PDOST", "PDOST_N", "PDOST_M"])


