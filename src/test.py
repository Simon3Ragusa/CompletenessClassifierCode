import torch
from hyperimpute.plugins.imputers import Imputers

from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

from xgbimputer import XGBImputer

from catboost import CatBoostRegressor, CatBoostClassifier

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
class impute_soft_imputer():
    def __init__(self):
        self.name = 'Soft Imputer'

    def fit(self, df):
        # X_incomplete_normalized = BiScaler().fit_transform(df)
        df = np.array(df).reshape(-1,1)
        df = pd.DataFrame(SoftImpute(verbose=False).fit_transform(df))
        return df

# Works with both types of features
class impute_xgb_imputer():
    def __init__(self):
        self.name = 'XGB Imputer'

    def fit(self, df, categorical_features_index, replace_values_back=True):
        imputer = XGBImputer(categorical_features_index=categorical_features_index, replace_categorical_values_back=replace_values_back)
        df = np.array(df)
        # print("We are inside the class. Input shape: ", df.shape)
        df = pd.DataFrame(imputer.fit_transform(df))
        return df
    
class impute_catboost():
    def __init__(self):
        self.name = 'CatBoost Imputer'

    def fit(self, df, missing_column):

        type_missing = df.dtypes[missing_column]
        #missing_column = df[missing_column]
        print("Type missing: ", type_missing)
        X = df.copy()

        # Select categorical features from the dataset
        cat_features = list(df.select_dtypes(include=["object", "bool"]).columns)

        if type_missing in ["int64", "float64"]:
            # Use CatBoostRegressor
            fully_available_samples = X[X[missing_column].notnull()]
            missing = X[X[missing_column].isnull()]

            X_train = fully_available_samples.drop(columns = [missing_column])
            print("X_train type: ", type(X_train))
            print("X_train shape: ", X_train.shape)
            y_train = fully_available_samples[missing_column]

            X_pred = missing.drop(columns = [missing_column])

            # Up to here we have the training set in X_train and y_train and the uncomplete samples in X_pred

            imputer = CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                loss_function='RMSE',
                verbose=False,
                random_seed=42
            )

            if len(fully_available_samples) > 1 and len(missing) > 0:
                imputer.fit(X_train, y_train, cat_features=cat_features)
                print(type(df))
                df.loc[df[missing_column].isnull(), missing_column] = imputer.predict(X_pred)
                df = pd.DataFrame(df)
                return df
            
            df = pd.DataFrame(columns=df.columns)
            return df
            
        elif type_missing in ["bool", "object"]:
            # Use CatBoostClassifier
            print("Debug")
            return 0


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
imputer_soft = impute_soft_imputer()

imputer_xgb = impute_xgb_imputer()

imputer_catboost = impute_catboost()

# Here we simulate an iteration on a list of datasets with missing values in the selected column, and we impute them one by one with
# different techniques

for i in range(len(df_list_no_class)):
    print("Column with missing values: ", column_to_inject_missing)
    print(type(column_to_inject_missing))
    column_type = df_list_no_class[i][column_to_inject_missing].dtype

    print("Column type: ", column_type)
    if column_type in ["int64", "float64"]:
        print("Imputation with Catboost Imputer - Missing percentage: ", round(df_list_no_class[i][column_to_inject_missing].isnull().sum()/df_list_no_class[i].shape[0],2))
        df_missing = df_list_no_class[i]
        # df_missing[class_name] = df[class_name]
        df_imputed_xgb = imputer_catboost.fit(df_missing, column_to_inject_missing)
        print(df_imputed_xgb.head())





