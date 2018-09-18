import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm as cm
matplotlib.use("PDF")
import pylab as py
from sklearn.decomposition import PCA
from PlayoffData import PlayoffData
import os

# -------------------------------------------------------------------------------------------------------
# PARAMETERS
# -------------------------------------------------------------------------------------------------------

# Input directory containing prepared data for each season
input_root_dir = "./data/prepared/"
input_template = "seriesYEAR_REPLACE.dat"

# Output directory containing data transformed by PCA
output_root_dir = "./data/pca/"
output_template = input_template

# NHL seasons to aggregate data from 
seasons = np.arange(2008, 2018+1)

# Flag to split training, validation, and testing data by season
USE_SEASON_SPLIT = False
nseasons_test    = 3

# Training split fraction and seed
train_fraction = 0.7
seed_split     = 3141592

# -------------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------------

def PCAColumnMapper(N):
    """
    Return dictionary that renames columns from 0, 1, ..., N to PC0, PC1, ..., PCN.
    """

    d = { }
    for i in range(N):
        d[i] = "PC" + str(i)

    return d

def SaveData(df, ddir, dtemp, year):
    """
    Save dataframe to CSV file.
    """

    ofile = ddir + dtemp.replace("YEAR_REPLACE", str(year))
    df.to_csv(ofile, na_rep="NA", sep="\t")
    print("Data saved to " + ofile)

# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------

#
# Create output directory if it does not yet exist
#

if not os.path.isdir(output_root_dir): os.makedirs(output_root_dir)

#
# Gather playoff data over all seasons
#

data = PlayoffData(input_root_dir, input_template, seasons)

#
# Perform PCA using only the training data
#

pca = PCA()
data.features = pd.DataFrame(pca.fit(data.x_train).transform(data.features))

#
# Write out each season data in terms of PCA transformation
#

# Rename columns
data.features.rename(PCAColumnMapper(data.num_features), axis="columns", inplace=True)

# Put metadata and target back in
df = pd.concat([data.metadata,data.features,data.target], axis=1)

for season in seasons:

    ds = df.loc[df.Season==season]
    SaveData(ds, output_root_dir, output_template, season)

