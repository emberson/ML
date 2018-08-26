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
    meta_columns = ["Team1", "Team2", "Round", "Matches", "Nlast"]

    # Value to initialize for random seeds
    seed_initial = 135792468

    def __init__(self, data_root_dir, template_file_name, seasons):
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

        # Perform feature scaling
        self.FeatureScaling()

        # Store useful information
        self.num_samples  = self.features.shape[0]
        self.num_features = self.features.shape[1]

        # Initialize random seeds which can be adjusted manually
        self.seed_split = self.seed_initial

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
        Normalize all floating point data to have zero mean and unit variance.
        Store original summary statistics (min, max, mean, variance) of the unscaled data in
        a dataframe.
        """

        keys = { }

        for column in self.features: 

            # Get summary stats
            x = self.features[column]
            xmin  = x.min() 
            xmax  = x.max()
            xmean = x.mean()
            xstd  = x.std()

            # Store summary stats of this feature
            keys[column] = np.array([xmin, xmax, xmean, xstd])

            # Perform feature scaling only on float data
            if x.dtype == float:
                self.features[column] = (self.features[column] - xmean)/xstd

        # Construct dataframe containing summary stats of unscaled data
        self.scaling = pd.DataFrame.from_dict(keys)
        self.scaling.rename(index={0:"min", 1:"max", 2:"mean", 3:"std"}, inplace=True)

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
        self.num_features = self.features.shape[1]

    def SetSplitSeed(self, seed):
        """
        Specify the seed for the training and testing split
        """

        self.seed_split = seed

    def SplitTrainingTestingData(self, frac_train):
        """
        Randomly splits the data so that frac_train of it is training and (1-frac_train) is testing.
        """

        # Split data and store in "full" dataframes which contain all feature vectors
        self.x_train_full, self.x_test_full, self.y_train, self.y_test = \
            train_test_split(self.features, self.target, test_size=1-frac_train, train_size=frac_train, random_state=self.seed_split)

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


