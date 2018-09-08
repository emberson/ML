import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

class PlayoffData:
    """
    Class that holds the prepared playoff data and performs all of the data manipulations
    (e.g., handle missing values, feature scaling, data splitting) necessary for each model.
    """

    # Column name containing target prediction
    target_column = "Result"

    # Columns containing playoff metadata which will be stored in a separate dataframe
    meta_columns = ["Team1", "Team2", "Round", "Season", "Matches", "Nlast"]

    # Default training-testing split seed 
    seed_default = 135792468

    # Default training split fraction
    train_fraction_default = 0.7

    def __init__(self, data_root_dir, template_file_name, seasons, seed_split=None, train_fraction=None, SeasonSplit=False, nSeasonTest=0):
        """
        Instantiate class by constructing the dataframes that hold each component of
        the playoff data (metadata, features, target, scaling)
        """

        # Read all of the prepared data and store in a single dataframe
        df = self.GatherAllData(data_root_dir, template_file_name, seasons)

        # Extract meta data, features, and target into separate dataframes
        self.metadata = df[self.meta_columns]
        self.features = df.drop(columns=self.meta_columns+[self.target_column], inplace=False)
        self.target   = df[self.target_column] 

        # Handle missing values
        self.HandleMissingValues()

        # Initialize training-testing split seed and fraction
        if seed_split is None: self.seed_split = self.seed_default
        else: self.seed_split = seed_split
        if train_fraction is None: self.train_fraction = self.train_fraction_default
        else: self.train_fraction = train_fraction
        self.season_split = SeasonSplit
        self.nseason_test = nSeasonTest

        # Perform training-testing split
        self.SplitTrainingTestingData()

        # Perform feature scaling based on training features
        self.FeatureScaling()

        # Store useful information
        self.num_samples  = self.features.shape[0]
        self.num_features = self.features.shape[1]

    def GatherAllData(self, ddir, dfile, seasons):
        """
        Read each individual season and accumulate into one dataframe
        """

        for i in range(seasons.shape[0]):
            dfi = self.ReadPreparedData(ddir, dfile, seasons[i])
            if i == 0: df = dfi.copy()
            else: df = pd.concat([df,dfi], ignore_index=True)

        return df

    def ReadPreparedData(self, ddir, dfile, year, rep_string="YEAR_REPLACE"):
        """
        Read dataframe containing prepared playoff series data. 
        """

        ff = ddir + dfile.replace(rep_string, str(year))
        df = pd.read_csv(ff, index_col=0, sep="\t")

        return df
    
    def HandleMissingValues(self):
        """
        Fill any missing values with mean of that column. 
        """

        imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
        imp.fit(self.features)
        self.features = pd.DataFrame(data=imp.transform(self.features), columns=self.features.columns)

    def FeatureScaling(self):
        """
        Normalize all floating point data to have zero mean and unit variance (based on the training set).
        Store original summary statistics (min, max, mean, variance) of the unscaled data in
        a dataframe.
        """

        keys = { }

        for column in self.features: 

            # Don't do this for categorical data
            if column == "Home": continue 

            # Get summary stats
            x = self.x_train_full[column]
            xmin  = x.min() 
            xmax  = x.max()
            xmean = x.mean()
            xstd  = x.std()

            # Store summary stats of this feature
            keys[column] = np.array([xmin, xmax, xmean, xstd])

            # Perform feature scaling only on float data
            if self.features[column].dtype == float:
                self.features[column]     = (self.features[column] - xmean)/xstd

        # Construct dataframe containing summary stats of unscaled data
        self.scaling = pd.DataFrame.from_dict(keys)
        self.scaling.rename(index={0:"min", 1:"max", 2:"mean", 3:"std"}, inplace=True)

        # Call this again to propogate normalization changes over to training and testing sets
        self.SplitTrainingTestingData()

    def ComputeInformationValue(self, nbins=5):
        """
        Compute the information value for each feature.
        """

        # Create temporary dataframe that holds features and target
        df = self.features.assign(target=self.target)

        self.IV = { }
        for column in self.features:
            if df[column].dtype == int: # Categorical
                db = df.groupby([column])["target"].agg(["count", "sum"])
            else:
                df["temp"] = pd.qcut(df[column].rank(method="first"), nbins)
                db = df.groupby(["temp"])["target"].agg(["count", "sum"])
            db.columns = ["T","W"]
            db["L"] = db["T"] - db["W"]
            db["WP"] = db["W"]/db["W"].sum()
            db["LP"] = db["L"]/db["L"].sum()
            db["WOE"] = np.log(db["WP"]/db["LP"])
            db["IV"] = (db["WP"] - db["LP"])*db["WOE"]
            self.IV[column] = db["IV"].sum()

    def SummarizeFeatures(self, sl=20):
        """
        Prints information on the ranges of the unscaled feature data. Useful for testing.
        """

        slt = 5*sl
        print("-"*slt)
        print("SUMMARY OF INPUT FEATURES".center(slt))
        print("-"*slt)
        print("Feature".ljust(sl) + "Min".ljust(sl) + "Max".ljust(sl) + "Mean".ljust(sl) + "Std".ljust(sl))
        for column in self.scaling:
            print(str(column).ljust(sl) + str(self.scaling[column]["min"]).ljust(sl) + str(self.scaling[column]["max"]).ljust(sl) + \
                  str(self.scaling[column]["mean"]).ljust(sl) + str(self.scaling[column]["std"]).ljust(sl))
        print("-"*slt)

    def KeepFeatures(self, columns):
        """
        Drop all but the selected features.
        """

        self.features = self.features[columns]
        self.num_features = self.features.shape[1]

    def DropFeatures(self, columns):
        """
        Drop features from consideration.
        """

        self.features.drop(columns=columns, inplace=True)
        self.x_train_full.drop(columns=columns, inplace=True)
        self.x_test_full.drop(columns=columns, inplace=True)
        self.x_train.drop(columns=columns, inplace=True)
        self.x_test.drop(columns=columns, inplace=True)
        self.num_features = self.features.shape[1]

    def SetSplitSeed(self, seed):
        """
        Specify the seed for the training and testing split
        """

        self.seed_split = seed

    def SplitTrainingTestingData(self):
        """
        Randomly splits the data so that train_fraction of it is training and (1-train_fraction) is testing.
        """

        if self.season_split:

            # Last self.nseason_test seasons are used for testing, others are training
            seasons = np.sort(np.unique(self.metadata.Season.tolist()))
            seasons_train = seasons[:seasons.shape[0]-self.nseason_test]
            seasons_test  = seasons[-self.nseason_test:]

            # Pull out the full training set and test set
            self.x_train_full = self.features.loc[self.metadata.Season.isin(seasons_train)]
            self.y_train = self.target.loc[self.metadata.Season.isin(seasons_train)]
            self.x_test_full = self.features.loc[self.metadata.Season.isin(seasons_test)]
            self.y_test = self.target.loc[self.metadata.Season.isin(seasons_test)]

            # Record how many seasons were used in each set
            self.num_seasons_train = seasons_train.shape[0]
            self.num_seasons_test  = seasons_test.shape[0]

        else:

            # Split data and store in "full" dataframes which contain all feature vectors
            self.x_train_full, self.x_test_full, self.y_train, self.y_test = \
                train_test_split(self.features, self.target, test_size=1-self.train_fraction, \
                                 train_size=self.train_fraction, shuffle=True, random_state=self.seed_split)
#                                 train_size=self.train_fraction, shuffle=True, stratify=self.metadata.Season, random_state=self.seed_split)

        # Initialize actual training and testing set to use the full set of features
        self.UseAllTrainingTestingFeatures()

    def UseAllTrainingTestingFeatures(self):
        """
        Set the training and testing sets to use all features.
        """

        self.x_train = self.x_train_full.copy()
        self.x_test  = self.x_test_full.copy()

    def UseSelectedTrainingTestingFeatures(self, features):
        """
        Set the training and testing sets to use only the selected features.
        """

        self.x_train = self.x_train_full[features]
        self.x_test  = self.x_test_full[features]

    def GrabMetaData(self, samples):
        """
        Returns metadata for each of the samples in sample.
        """

        return self.metadata.iloc[samples.index.tolist()]

