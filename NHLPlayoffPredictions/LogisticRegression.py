import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm as cm
matplotlib.use("PDF")
import pylab as py
from sklearn import model_selection
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from PlayoffData import PlayoffData
import os

# https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

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

# Kfold parameters
seed_kfold   = 78
nsplit_kfold = 3

# Various plots
plot_dir = "plots/LogisticRegression/"
plot_correlation_mat = plot_dir + "correlation_matrix.pdf"
plot_nfeatures_tverr = plot_dir + "nfeatures_error.pdf" 
plot_roc_curve       = plot_dir + "roc.pdf"

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

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

def PlotError(pf, tacc, vacc):
    """
    Make a plot comparing the training and validation error as a function of the feature set size.
    """

    #
    # Plot paramters
    #

    # Labels
    xlabel = r"$N_{\rm features}$"
    ylabel = r"${\rm Error}$"
    tlabel = r"${\rm Training}$"
    vlabel = r"${\rm Validation}$"

    # Plot window
    left   = 0.11
    right  = 0.97
    bottom = 0.11
    top    = 0.97

    #
    # Prepare the data
    #

    nf = np.arange(tacc.shape[0])+1

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

    py.plot(nf, 1-tacc, "k-", label=tlabel)
    py.plot(nf, 1-vacc, "r-", label=vlabel)

    lg = py.legend(loc="best", fancybox=True)
    lg.draw_frame(False)

    py.xlabel(xlabel)
    py.ylabel(ylabel)

    py.savefig(pf)
    py.close()

    print("Error plot saved to " + pf)

def PlotROCCurve(pf, fpr, tpr, auc): 
    """
    Plot ROC curve for logistic regression model.
    """

    #
    # Plot parameteres
    #

    # Labels
    xlabel = r"${\rm FPR}$"
    ylabel = r"${\rm TPR}$"
    title  = r"${\rm AUC\ =\ " + str(auc) + "}$"

    # Ranges
    xmin = 0.
    xmax = 1.
    ymin = 0.
    ymax = 1.

    # Plot window
    left   = 0.15
    right  = 0.97
    bottom = 0.11
    top    = 0.93

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

    ax1 = py.subplot(111)

    py.plot(fpr, tpr, "r-") 
    py.plot([xmin, xmax], [xmin, xmax], "k:")

    py.title(title)
    py.xlabel(xlabel)
    py.ylabel(ylabel)

    ax1.set_xlim((xmin, xmax))
    ax1.set_ylim((ymin, ymax))

    py.savefig(pf)
    py.close()

    print("ROC plot saved to " + pf)

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

# Make output plot directory if it does not yet exist
if not os.path.isdir(plot_dir): os.makedirs(plot_dir)

#
# Gather playoff data over all seasons
#

data = PlayoffData(input_root_dir, input_template, seasons)
#data.DropFeatures(["Home", "BB", "CD", "CD_M", "CD_N", "GD", "GD_N", "GD_M", "GDST", "GDST_M", "GDST_N"]) 

#
# Plot correlation matrix of the feature set
#

PlotCorrelationMatrix(plot_correlation_mat, data.features.corr())

#
# Compute information value for each feature
#

data.ComputeInformationValue()

# 
# Split data between training and testing samples
#

data.SetSplitSeed(seed_split)
data.SplitTrainingTestingData(train_fraction)

#
# Use recursive feature elimination to rank the importance of the variables
#

logreg = LogisticRegression(penalty="l2", C=1)
rfe = RFE(logreg, 1)
rfe = rfe.fit(data.x_train, data.y_train)
rank = rfe.ranking_ - 1

print(50*"-")
print("Recursive feature elimation ranking".upper())
print(50*"-")
cols = np.array(data.features.columns.tolist())[np.argsort(rank)]
for i in range(rank.shape[0]):
    print(str(i).ljust(4) + " : " + str(cols[i]).ljust(20) + str(data.IV[cols[i]]).ljust(20))
print(50*"-")

#
# Compute training and validation accuracy as a function of the number of retained variables
#

nfeatures = data.num_features
train_acc = np.zeros(nfeatures, dtype="float64")
valid_acc = np.zeros(nfeatures, dtype="float64")

kfold = model_selection.KFold(n_splits=nsplit_kfold, random_state=seed_kfold)

for i in range(nfeatures):

    # Restrict data to only the desired set of features
    cols = data.features.columns[np.where(rank <= i)] 
    data.UseSelectedTrainingTestingFeatures(cols)

    # Run logistic regression with reduced data set
    logreg.fit(data.x_train, data.y_train) 

    # Use K-fold cross validation to compute training and validation accuracy 
    result = model_selection.cross_validate(logreg, data.x_train, y=data.y_train, cv=kfold, scoring="accuracy", return_train_score=True)
    train_acc[i] = result["train_score"].mean()
    valid_acc[i] = result["test_score"].mean()

#
# Make a plot comparing training and validation accuracy as a function of number of retained variables
#

PlotError(plot_nfeatures_tverr, train_acc, valid_acc)

#
# Choose the number of features as that which maximizes the validation accuracy
#

nf_use   = valid_acc.argmax()+1
cols_use = data.features.columns[np.where(rank < nf_use)]
print("Choosing to use " + str(nf_use) + " features")
print("Training accuracy   : {:.2f}".format(train_acc[nf_use-1]))
print("Validation accuracy : {:.2f}".format(valid_acc[nf_use-1]))

#
# Evaluate test accuracy 
#

data.UseSelectedTrainingTestingFeatures(cols_use)
logreg.fit(data.x_train, data.y_train)
result = model_selection.cross_validate(logreg, data.x_train, y=data.y_train, cv=kfold, scoring="accuracy", return_train_score=True)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(data.x_test, data.y_test)))

#
# Plot ROC curve
#

fpr, tpr, eps = roc_curve(data.y_test, logreg.predict_proba(data.x_test)[:,1])
auc = roc_auc_score(data.y_test, logreg.predict(data.x_test))
PlotROCCurve(plot_roc_curve, fpr, tpr, auc)

#
# Print confusion matrix for prediciton
#

y_pred = logreg.predict(data.x_test)
confusion_matrix = confusion_matrix(data.y_test, y_pred)
print(confusion_matrix)
print(classification_report(data.y_test, y_pred))
print("\nC-statistic:",round(auc,4))

#
# Compute VIF for each feature 
#

print("Variance Inflation Factor")
cnames = data.x_train.columns
for i in np.arange(0,len(cnames)):
    xvars = list(cnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(data.x_train[yvar], sm.add_constant(data.x_train[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared)
    print (yvar,round(vif,3))

