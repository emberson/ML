import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("PDF")
import pylab as py
from sklearn.model_selection import KFold,GridSearchCV,cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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
max_features0      = "auto"
bootstrap0         = True
n_jobs0            = -1
seed0              = 299792458

# Hyperparameters to optimize
HYPERPARAMETER_SEARCH    = True
n_estimators_search      = np.array([50, 75, 100, 125])
max_depth_search         = np.array([1, 2, 3, 4])
min_samples_split_search = np.array([2, 4, 8, 16])
min_samples_leaf_search  = np.array([1, 2, 4, 6])

# If skipping the optimization step, use these parameters instead
n_estimators0      = 100
max_depth0         = 3
min_samples_split0 = 2
min_samples_leaf0  = 1

# Various plots
plot_dir = "plots/RandomForest/"
plot_feature_summary = plot_dir + "feature-F1-F2.pdf"

# Some flags
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
# Drop multicollinear features 
#

data.DropFeatures(["Home", "BB", "CD", "CD_N", "CD_M", "PWP", "PDO", "PDO_N", "PDO_M", "PDOST", "PDOST_N", "PDOST_M"])

#
# Setup random forest classifier and kfold validation model
#

rf = RandomForestClassifier(criterion=criterion0, max_features=max_features0, bootstrap=bootstrap0, random_state=seed0, n_jobs=n_jobs0)

if USE_SEASON_SPLIT:
    kfold = KFold(n_splits=data.num_seasons_train, shuffle=False)
else:
    kfold = KFold(n_splits=nsplit_kfold, random_state=seed_kfold)

#
# Use grid search to optimize hyperparameters
#

if HYPERPARAMETER_SEARCH:

    search_params = {"n_estimators":n_estimators_search, "max_depth":max_depth_search, \
                     "min_samples_split":min_samples_split_search, "min_samples_leaf":min_samples_leaf_search}

    grid = GridSearchCV(rf, search_params, cv=kfold, scoring="accuracy", return_train_score=True)
    grid.fit(data.x_train, y=data.y_train)

    # Unpack optimal parameters 
    opt_params         = grid.best_params_
    n_estimators0      = opt_params["n_estimators"]
    max_depth0         = opt_params["max_depth"]
    min_samples_split0 = opt_params["min_samples_split"]
    min_samples_leaf0  = opt_params["min_samples_leaf"]

#
# Evaluate accuracy with optimal parameters
#

rf = RandomForestClassifier(n_estimators=n_estimators0, max_depth=max_depth0, min_samples_split=min_samples_split0, min_samples_leaf=min_samples_leaf0, \
                            criterion=criterion0, max_features=max_features0, bootstrap=bootstrap0, random_state=seed0, n_jobs=n_jobs0)

rf.fit(data.x_train, data.y_train)
result = cross_validate(rf, data.x_train, y=data.y_train, cv=kfold, scoring="accuracy", return_train_score=True)
print("Training accuracy   : {:.2f} +/- {:.2f}".format(result["train_score"].mean(), result["train_score"].std()))
print("Validation accuracy : {:.2f} +/- {:.2f}".format(result["test_score"].mean(), result["test_score"].std()))
print("Test accuracy       : {:.2f}".format(rf.score(data.x_test, data.y_test)))

#
# Print confusion matrix for prediciton
#

y_pred = rf.predict(data.x_test)
confusion_matrix = confusion_matrix(data.y_test, y_pred)
PrintConfusionMatrix(confusion_matrix)

#
# Print information regarding misclassified samples and make plots showing where errors appear in feature space
#

metadata = data.GrabMetaData(data.x_test)
PrintErrorSummary(metadata, data.x_test, data.y_test, y_pred)
if PLOT_FEATURE_SUMMARY:
    for i in range(cols_use.shape[0]):
        plot_file_1 = plot_feature_summary.replace("F1", str(cols_use[i]))
        for j in range(i+1, cols_use.shape[0]):
            plot_file = plot_file_1.replace("F2", str(cols_use[j]))
            PlotFeatureSummary(plot_file, data.x_test, cols_use[i], cols_use[j], data.y_test, y_pred)

#
# Show final ranking of importance
#

scores = rf.feature_importances_
isort = np.argsort(scores)[::-1]
rank = np.zeros(len(data.x_train.columns.tolist()), dtype="int32")
for i in range(rank.shape[0]): rank[i] = np.where(isort == i)[0]
scores = scores[isort] 

print(50*"-")
print("Feature ranking".upper())
print(50*"-")
cols = np.array(data.x_train.columns.tolist())[np.argsort(rank)]
for i in range(rank.shape[0]):
    print(str(i).ljust(4) + " : " + str(cols[i]).ljust(20) + str(scores[i]).ljust(20))
print(50*"-")

