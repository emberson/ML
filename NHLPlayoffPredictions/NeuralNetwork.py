import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("PDF")
import pylab as py
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from keras import models
from keras import layers
from keras import regularizers
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

# Hyperparameters to optimize
HYPERPARAMETER_SEARCH = False
nunits_search         = np.array([12, 16, 20, 24])
nlayers_search        = np.array([1, 2])
l1_search             = np.array([0.01, 0.001, 0.0001])
dropout_search        = np.array([0.2, 0.3, 0.4, 0.5])

# If skipping the optimization step, use these parameters instead (found by doing the search)
nepochs0 = 30
nunits0  = 20
nlayers0 = 2
l10      = 0.0001
dropout0 = 0.3

# Various plots
plot_dir = "plots/NeuralNetwork/"
plot_nepochs_tverr = plot_dir + "nepochs_error.pdf"

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

def PrintConfusionMatrix(cmat, sh=50, sl=15):
    """

dropout_search        = np.array([0.4, 0.5, 0.6])
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

def PlotError(pf, xx, tacc, vacc):
    """
    Make a plot comparing the training and validation error as a function of the epoch count. 
    """

    #
    # Plot paramters
    #

    # Labels
    xlabel = r"$N_{\rm epoch}$" 
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
# Setup kfold cross validation model
#

if USE_SEASON_SPLIT:
    kfold = KFold(n_splits=data.num_seasons_train, shuffle=False)
else:
    kfold = KFold(n_splits=nsplit_kfold, random_state=seed_kfold, shuffle=True)

nfold = kfold.get_n_splits(data.x_train)

#
# Perform hyperparameter grid search
#

if HYPERPARAMETER_SEARCH:

    results = np.array([ ], dtype="float64")
    params  = { }
    niter   = 0

    # Only allow for a maximum of two layers
    assert nlayers_search.max() < 3

    for nunit in nunits_search:
        for nlayer in nlayers_search:
            for l1 in l1_search:
                for dropout in dropout_search:

                    # Build the neural network
                    model = models.Sequential()
                    model.add(layers.Dense(nunit, activation="relu", kernel_regularizer=regularizers.l1(l1), input_shape=(data.x_train.shape[1],)))
                    model.add(layers.Dropout(dropout))
                    if nlayer == 2:
                        model.add(layers.Dense(nunit, activation="relu", kernel_regularizer=regularizers.l1(l1)))
                        model.add(layers.Dropout(dropout))
                    model.add(layers.Dense(1, activation="sigmoid"))
                    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

                    # Compute kfold cross-validation accuracy
                    valid_acc = np.array([ ], dtype="float64")
                    for tinds, vinds in kfold.split(data.x_train):
                        x_train, x_val = data.x_train.iloc[tinds], data.x_train.iloc[vinds]
                        y_train, y_val = data.y_train.iloc[tinds], data.y_train.iloc[vinds]
                        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=nepochs0, batch_size=1, verbose=0)
                        valid_acc = np.append(valid_acc, history.history["val_acc"])
                    valid_acc = valid_acc.reshape(nfold, nepochs0).mean(axis=0)

                    # Store max accuracy in array
                    results = np.append(results, valid_acc.max())

                    # Store hyperparameters in dictionary
                    params[niter] = {"nunit": nunit, "nlayer": nlayer, "l1": l1, "dropout": dropout} 

                    # Print some info to screen
                    print(params[niter], " validation accuracy: ", results[niter])

                    # Update counter 
                    niter += 1

    # Set optimal hyperparameters as those which maximize validation accuracy
    niter0   = np.argmax(results)
    params0  = params[niter0]
    nunits0  = params0["nunit"]
    nlayers0 = params0["nlayer"]
    l10      = params0["l1"]
    dropout0 = params0["dropout"]

#
# Retrain optimized model (use defaults if skipping HYPERPARAMETER_SEARCH) 
#

print("Training optimal neural network with the following hyperparameters")
print("nlayers : ", nlayers0)
print("nunits  : ", nunits0)
print("l1      : ", l10)
print("dropout : ", dropout0)

train_acc_history = np.array([ ], dtype="float64") 
valid_acc_history = np.array([ ], dtype="float64")

for tinds, vinds in kfold.split(data.x_train):

    # Extract training and validation data
    x_train, x_val = data.x_train.iloc[tinds], data.x_train.iloc[vinds]
    y_train, y_val = data.y_train.iloc[tinds], data.y_train.iloc[vinds]

    # Build the neural network
    model = models.Sequential()
    model.add(layers.Dense(nunits0, activation="relu", kernel_regularizer=regularizers.l1(l10), input_shape=(data.x_train.shape[1],)))
    model.add(layers.Dropout(dropout0))
    if nlayers0 == 2:
        model.add(layers.Dense(nunits0, activation="relu", kernel_regularizer=regularizers.l1(l10)))
        model.add(layers.Dropout(dropout0))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=nepochs0, batch_size=1, verbose=0)

    # Append results to global lists
    train_acc_history = np.append(train_acc_history, history.history["acc"])
    valid_acc_history = np.append(valid_acc_history, history.history["val_acc"])

train_acc_history = train_acc_history.reshape(nfold, nepochs0)
valid_acc_history = valid_acc_history.reshape(nfold, nepochs0)

#
# Make a plot comparing training and validation accuracy as a function of epoch number 
#

train_acc = train_acc_history.mean(axis=0)
train_std = train_acc_history.std(axis=0)
valid_acc = valid_acc_history.mean(axis=0)
valid_std = valid_acc_history.std(axis=0)

PlotError(plot_nepochs_tverr, np.arange(nepochs0)+1, train_acc, valid_acc)

#
# Evaluate test accuracy 
#

test_loss, test_acc = model.evaluate(data.x_test, data.y_test)
print("Training accuracy   : {:.2f} +/- {:.2f}".format(train_acc[-1], train_std[-1]))
print("Validation accuracy : {:.2f} +/- {:.2f}".format(valid_acc[-1], valid_std[-1]))
print("Test accuracy       : {:.2f}".format(test_acc))

#
# Print confusion matrix for prediciton
#

y_pred = model.predict_classes(data.x_test) ; y_pred = y_pred.reshape(np.prod(y_pred.shape))
confusion_matrix = confusion_matrix(data.y_test, y_pred)
PrintConfusionMatrix(confusion_matrix)

#
# Print information regarding misclassified samples
#

metadata = data.GrabMetaData(data.x_test)
PrintErrorSummary(metadata, data.x_test, data.y_test, y_pred)

