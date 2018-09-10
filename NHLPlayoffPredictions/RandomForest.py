import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm as cm
matplotlib.use("PDF")
import pylab as py
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE,SelectKBest
from PlayoffData import PlayoffData
import os

# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Input directory containing prepared data for each season
input_root_dir = "./data/prepared/"
input_template = "seriesYEAR_REPLACE.dat"

# NHL seasons to aggregate data from 
seasons = np.arange(2008, 2018+1)

# Flag to split training, validation, and testing data by season
USE_SEASON_SPLIT = False
nseasons_test    = 3 

# Training split fraction and seed
train_fraction = 0.7
seed_split     = 3141592

# Kfold parameters
seed_kfold   = 2718281
nsplit_kfold = 5 

# Default parameters for random forest classifier
criterion0         = "gini"
n_estimators0      = 500
max_depth0         = 25 
min_samples_split0 = 3
min_samples_leaf0  = 4

# Various plots
plot_dir = "plots/RandomForest/"
plot_nfeatures_tverr = plot_dir + "nfeatures_error.pdf" 
plot_reg_tverr       = plot_dir + "regularization_error.pdf" 
plot_feature_summary = plot_dir + "feature-F1-F2.pdf"

# Some flags
USE_RFE              = False
PLOT_FEATURE_SUMMARY = False

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

def PrintConfusionMatrix(cmat, sh=50, sl=15):
    """
    Print confusion matrix to screen along with details of incorrectly classified data (if desired). 
    """

    # Print the confusion matrix to screen
    tn, fp, fn, tp = cmat.ravel()
    print(sh*"-")
    print("Model Summary".upper())
    print(sh*"-")
    print(("TP = "+str(tp)).ljust(sl) + ("FP = "+str(fp)).ljust(sl))
    print(("FN = "+str(fn)).ljust(sl) + ("TN = "+str(tn)).ljust(sl))
    precision = tp/(1.*(tp+fp))
    recall    = tp/(1.*(tp+fn))
    f1        = 2*precision*recall/(precision+recall)
    print("P   : {:.2f}".format(precision))
    print("R   : {:.2f}".format(recall))
    print("F1  : {:.2f}".format(f1))
    print(sh*"-")

def PrintErrorSummary(m0, x0, y0, yp, sh=50):
    """
    Print false positives and false negatives to screen.
    """

    print(sh*"=")
    print("FALSE POSITIVES".center(sh))
    ip = np.where((y0 == 0) & (yp == 1))
    mp = m0.iloc[ip]
    fp = x0.iloc[ip]
    print(pd.concat([mp,fp], axis=1))
    print(sh*"=")
    print("FALSE NEGATIVES".center(sh))
    jn = np.where((y0 == 1) & (yp == 0))
    mn = m0.iloc[jn]
    fn = x0.iloc[jn]
    print(pd.concat([mn,fn], axis=1))
    print(sh*"=")

def PlotError(pf, xx, tacc, vacc, command):
    """
    Make a plot comparing the training and validation error as a function of the feature set size.
    """

    #
    # Plot paramters
    #

    # Labels
    if command == 0:
        xlabel = r"$N_{\rm features}$"
    elif command == 1:
        xlabel = r"${\rm log}_{10}(C)$"
    ylabel = r"${\rm Error}$"
    tlabel = r"${\rm Training}$"
    vlabel = r"${\rm Validation}$"

    # Plot window
    left   = 0.11
    right  = 0.97
    bottom = 0.11
    top    = 0.97

    #
    # Make the plot
    #

    fig_width_pt  = 300.
    inches_per_pt = 1. / 72.27
    fig_width     = fig_width_pt * inches_per_pt
    fig_height    = fig_width
    fig_size      = [fig_width, fig_height]
    params        = {'backend': 'ps', 'axes.labelsize': 13, 'font.size': 12, \
                    'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12, \
                    'text.usetex': True, 'figure.figsize': fig_size}

    py.rcParams.update(params)
    fig = py.figure(1)
    py.clf()
    py.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=0., wspace=0.)

    py.plot(xx, 1-tacc, "k-", label=tlabel)
    py.plot(xx, 1-vacc, "r-", label=vlabel)

    lg = py.legend(loc="best", fancybox=True)
    lg.draw_frame(False)

    py.xlabel(xlabel)
    py.ylabel(ylabel)

    py.savefig(pf)
    py.close()

    print("Error plot saved to " + pf)

def LabelFMT(labs):
    """
    Formats labs to be compatible with latex.
    """

    labs = np.array(labs)
    for i in range(labs.shape[0]): labs[i] = labs[i].replace("_", "-")

    return labs

def PlotFeatureSummary(pf, x0, c1, c2, y0, yp):
    """
    Make a plot showing where in feature space false positives and false negatives occurred.
    """

    #
    # Plot paramters
    #

    # Labels
    xlabel = LabelFMT([c1])[0]
    ylabel = LabelFMT([c2])[0]

    # Plot window
    left   = 0.15
    right  = 0.97
    bottom = 0.11
    top    = 0.97

    #
    # Prepare the data
    #

    itp = np.where((y0 == 1) & (yp == 1))
    itn = np.where((y0 == 0) & (yp == 0))
    ifp = np.where((y0 == 0) & (yp == 1))
    ifn = np.where((y0 == 1) & (yp == 0))

    xtp = x0[c1].iloc[itp] ; ytp = x0[c2].iloc[itp]
    xtn = x0[c1].iloc[itn] ; ytn = x0[c2].iloc[itn]
    xfp = x0[c1].iloc[ifp] ; yfp = x0[c2].iloc[ifp]
    xfn = x0[c1].iloc[ifn] ; yfn = x0[c2].iloc[ifn]

    #
    # Make the plot
    #

    fig_width_pt  = 300.
    inches_per_pt = 1. / 72.27
    fig_width     = fig_width_pt * inches_per_pt
    fig_height    = fig_width
    fig_size      = [fig_width, fig_height]
    params        = {'backend': 'ps', 'axes.labelsize': 13, 'font.size': 12, \
                    'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12, \
                    'text.usetex': True, 'figure.figsize': fig_size}

    py.rcParams.update(params)
    fig = py.figure(1)
    py.clf()
    py.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=0., wspace=0.)

    py.plot(xtp, ytp, "go", label="FP")
    py.plot(xtn, ytn, "ro", label="FP")
    py.plot(xfp, yfp, "rx", label="FP")
    py.plot(xfn, yfn, "gx", label="FN")

    py.xlabel(xlabel)
    py.ylabel(ylabel)

    py.savefig(pf)
    py.close()

    print("Feature summary plot saved to " + pf)

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

# Make output plot directory if it does not yet exist
if not os.path.isdir(plot_dir): os.makedirs(plot_dir)

#
# Gather playoff data over all seasons
#

data = PlayoffData(input_root_dir, input_template, seasons, seed_split, train_fraction, USE_SEASON_SPLIT, nseasons_test)

#
# Compute information value for each feature
#

data.ComputeInformationValue()

#
# Rank importance of the variables
#

if USE_RFE:
    rf = RandomForestClassifier(n_estimators=n_estimators0, criterion=criterion0, max_depth=max_depth0, \
                                min_samples_split=min_samples_split0, min_samples_leaf=min_samples_leaf0)
    rfe = RFE(rf, 1)
    rfe = rfe.fit(data.x_train, data.y_train)
    rank = rfe.ranking_ - 1
else:
    select = SelectKBest(k=data.num_features)
    select_fit = select.fit(data.x_train, data.y_train)
    scores  = select_fit.scores_
    isort = np.argsort(scores)[::-1]
    rank = np.zeros(data.num_features, dtype="int32")
    for i in range(data.num_features): rank[i] = np.where(isort == i)[0]

print(50*"-")
print("Feature ranking".upper())
print(50*"-")
cols = np.array(data.features.columns.tolist())[np.argsort(rank)]
for i in range(rank.shape[0]):
    print(str(i).ljust(4) + " : " + str(cols[i]).ljust(20) + str(data.IV[cols[i]]).ljust(20))
print(50*"-")

#
# Setup default random forest classifier to determine number of retained variables
#

rf = RandomForestClassifier(n_estimators=n_estimators0, criterion=criterion0, max_depth=max_depth0, \
                            min_samples_split=min_samples_split0, min_samples_leaf=min_samples_leaf0)

#
# Compute training and validation accuracy as a function of the number of retained variables
#

nfeatures = data.num_features
train_acc = np.zeros(nfeatures, dtype="float64")
valid_acc = np.zeros(nfeatures, dtype="float64")
train_std = np.zeros(nfeatures, dtype="float64")
valid_std = np.zeros(nfeatures, dtype="float64")

if USE_SEASON_SPLIT:
    kfold = model_selection.KFold(n_splits=data.num_seasons_train, shuffle=False)
else:
    kfold = model_selection.KFold(n_splits=nsplit_kfold, random_state=seed_kfold)

for i in range(nfeatures):

    # Restrict data to only the desired set of features
    cols = data.features.columns[np.where(rank <= i)]
    data.UseSelectedTrainingTestingFeatures(cols)

    # Run logistic regression with reduced data set
    rf.fit(data.x_train, data.y_train)

    # Use K-fold cross validation to compute training and validation accuracy 
    result = model_selection.cross_validate(rf, data.x_train, y=data.y_train, cv=kfold, scoring="accuracy", return_train_score=True)
    train_acc[i] = result["train_score"].mean()
    valid_acc[i] = result["test_score"].mean()
    train_std[i] = result["train_score"].std()
    valid_std[i] = result["test_score"].std()

#
# Make a plot comparing training and validation accuracy as a function of number of retained variables
#

PlotError(plot_nfeatures_tverr, np.arange(nfeatures)+1, train_acc, valid_acc, 0)

#
# Choose the number of features as that which maximizes the validation accuracy
#

nf_use     = valid_acc.argmax()+1
cols_use   = data.features.columns[np.where(rank < nf_use)]
print("Choosing nfeatures  : {:d}".format(nf_use))
print("Training accuracy   : {:.2f} +/- {:.2f}".format(train_acc[nf_use-1], train_std[nf_use-1]))
print("Validation accuracy : {:.2f} +/- {:.2f}".format(valid_acc[nf_use-1], valid_std[nf_use-1]))
data.UseSelectedTrainingTestingFeatures(cols_use)

rf.fit(data.x_train, data.y_train)
print("Test accuracy       : {:.2f}".format(rf.score(data.x_test, data.y_test)))

#
# Show final ranking of importance
#

rfe = RFE(rf, 1)
rfe = rfe.fit(data.x_train, data.y_train)
rank = rfe.ranking_ - 1

print(50*"-")
print("Feature ranking".upper())
print(50*"-")
cols = np.array(data.x_train.columns.tolist())[np.argsort(rank)]
for i in range(rank.shape[0]):
    print(str(i).ljust(4) + " : " + str(cols[i]).ljust(20) + str(data.IV[cols[i]]).ljust(20))
print(50*"-")

