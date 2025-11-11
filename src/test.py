import torch
from hyperimpute.plugins.imputers import Imputers

from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

from xgbimputer import XGBImputer

import numpy as np
from Feature_selection.feature_selection import feature_selection_univariate, fixed_fs_univariate, remove_corr
from Column_profile_extraction.numerical import get_features_num
from Datasets.get_dataset import get_dataset
from Column_profile_extraction.categorical import get_features_cat
from Imputation.imputation_techniques import impute_missing_column
from Classification.algorithms_class import classification
from itertools import repeat
from multiprocessing import Pool
from utils import dirty_single_column, encoding_categorical_variables
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# Only numerical features
class impute_expectation_maximization():
    def __init__(self):
        self.name = 'Expectation Maximization'

    def fit(self, df):
        plugin = Imputers().get("EM")
        df = plugin.fit_transform(df)
        return df

# Only numerical features
class impute_soft_impute():
    def __init__(self):
        self.name = 'Soft Impute'

    def fit(self, df):
        # X_incomplete_normalized = BiScaler().fit_transform(df)
        df = np.array(df).reshape(-1,1)
        df = pd.DataFrame(SoftImpute(verbose=False).fit_transform(df))
        return df
    
class impute_xgb_imputer():
    def __init__(self):
        self.name = 'XGB Imputer'

    def fit(self, df):
        imputer = XGBImputer()
        df = imputer.fit_transform(df)
        return df
        

path_datasets = "Datasets/CSV/"
dataset = "abalone"
df = get_dataset(path_datasets,dataset + ".csv")

print("------------" + dataset + "------------")
df = get_dataset(path_datasets,dataset + ".csv")
class_name = df.columns[-1]

# feature selection
# df_fs, _, _, _, _ = feature_selection_univariate(df, class_name, perc_num=50, perc_cat=60)
df_corr_removed = remove_corr(df, class_name, threshold=0.8)
df_fs = fixed_fs_univariate(df_corr_removed, class_name)

columns = list(df_fs.columns)
columns.remove(class_name)

print("Columns selected for the experiments: " + str(columns))

column_to_inject_missing = columns[0]
# inject missing values in the df, with different percentages. This data frame contains different versions of the column with missing values (different percentages)
df_list_no_class = dirty_single_column(df[columns], column_to_inject_missing, class_name, 10)

# This contains different versions of the dataset with missing values in the selected column
print("Dataset before imputation: ", df_list_no_class[0].head())
print("Dataset shape before imputation: ", df_list_no_class[0].shape)


# Lets try imputation with EM: works on numerical columns only
imputer_em = impute_expectation_maximization()

# for each version of the dataset with missing values, impute the missing values in the selected column
imputer_soft = impute_soft_impute()

# Here we simulate an iteration on a list of datasets with missing values in the selected column, and we impute them one by one with
# different techniques

for i in range(len(df_list_no_class)):
    print("Column with missing values: ", column_to_inject_missing)
    column_type = df_list_no_class[i][column_to_inject_missing].dtype

    print("Column type: ", column_type)
    if column_type in ["int64", "float64"]:
        print("Imputation with soft impute - Missing percentage: ", round(df_list_no_class[i][column_to_inject_missing].isnull().sum()/df_list_no_class[i].shape[0],2))
        df_missing = df_list_no_class[i][column_to_inject_missing]
        # df_missing[class_name] = df[class_name]
        df_imputed_soft = imputer_soft.fit(df_missing)
        print(df_imputed_soft.head())





