import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
import torch
from hyperimpute.plugins.imputers import Imputers

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler

from xgbimputer import XGBImputer

from catboost import CatBoostRegressor, CatBoostClassifier

import miceforest as mf

from autoimpute.imputations import MiceImputer

os.environ["PYTENSOR_FLAGS"] = "cxx="

import numpy as np
from Feature_selection.feature_selection import feature_selection_univariate, fixed_fs_univariate, remove_corr
from Column_profile_extraction.numerical import get_features_num
from Datasets.get_dataset import get_dataset
from Column_profile_extraction.categorical import get_features_cat
from Imputation.imputation_techniques import impute_missing_column
from Classification.algorithms_class import classification
from itertools import repeat
from multiprocessing import Pool
from utils import dirty_single_column, encoding_categorical_variables, restore_nans
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
            cat_features = [feat for feat in cat_features if feat != missing_column]
            # Use CatBoostClassifier
            fully_available_samples = X[X[missing_column].notnull()]
            missing = X[X[missing_column].isnull()]

            # # encode categorical variables
            # fully_available_samples = encoding_categorical_variables(fully_available_samples)
            # print("Fully available samples after encoding: ", fully_available_samples.head())

            # missing = encoding_categorical_variables(missing)

            X_train = fully_available_samples.drop(columns = [missing_column])
            y_train = fully_available_samples[missing_column]

            X_pred = missing.drop(columns = [missing_column])

            imputer = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                loss_function='MultiClass',
                verbose=False,
                random_seed=42
            )

            if len(fully_available_samples) > 1 and len(missing) > 0:
                imputer.fit(X_train, y_train, cat_features=cat_features)
                df.loc[df[missing_column].isnull(), missing_column] = imputer.predict(X_pred)
                df = pd.DataFrame(df)
                return df

            print("Debug")
            return 0
        
# Both kind of features
class impute_rfi():
    def __init__(self):
        self.name = 'Random Forest Imputer'

    def fit(self, df, missing_column):
        # Get categorical features index
        categorical_features = list(df.select_dtypes(include=["object", "bool"]).columns)
        categorical_features_index = [df.columns.get_loc(col) for col in categorical_features]
        
        for col in df.select_dtypes(include=["object" ,"bool"]).columns:
            df[col] = df[col].astype('category')
        
        kernel = mf.ImputationKernel(
            data=df,
            save_all_iterations_data=True,
            random_state=42
        )

        kernel.mice(3)

        df = kernel.complete_data()
        return df

# To test
class impute_autoimpute():
    def __init__(self):
        self.name = 'Autoimpute'

    def fit(self, df):
        mice = MiceImputer(return_list=True)
        mice = mice.fit(df)
        df = pd.DataFrame(mice.transform(df))
        return df

# Both GAIN and VAE work with both types of features
class impute_gain():
    def __init__(self):
        self.name = 'GAIN Impute'

    def fit(self, df, missing_column):
        plugin = Imputers().get("gain")

        if df[missing_column].dtype in ["int64", "float64"]:
            # encode categorical variables
            X = df.copy()
            target = X[missing_column]
            X = X.drop(columns=[missing_column])
            X = encoding_categorical_variables(X)
            X[missing_column] = target
            X = X.astype(float)

            columns = list(X.columns)

            df = plugin.fit_transform(X)
            df = pd.DataFrame(df)

            # maps back the original columns
            df.columns = columns

            return df
        else:
            df = encoding_categorical_variables(df)
            df = restore_nans(df)
            df = df.astype(float)
            columns = list(df.columns)
            print("Data types after encoding: ", df.head())
            df = plugin.fit_transform(df)
            df = pd.DataFrame(df)
            df.columns = columns

            # Take the max of the "probabilities" to decide the final category, given the one hot encoding of the missing categorical variable
            col_prefix = missing_column + "_"
            targets = df.columns[df.columns.str.startswith(col_prefix)]
            print("Targets: ", targets)
            for index in df.index:
                max_value = -1
                selected_col = None
                for col in targets:
                    if df.loc[index, col] > max_value:
                        max_value = df.loc[index, col]
                        selected_col = col
                # Set all target columns to 0
                for col in targets:
                    df.loc[index, col] = 0
                # Set the selected column to 1
                if selected_col is not None:
                    df.loc[index, selected_col] = 1
            return df
        
class impute_vae():
    def __init__(self):
        self.name = 'VAE Impute'

    def fit(self, df, missing_column):
        plugin = Imputers().get("miwae")

        if df[missing_column].dtype in ["int64", "float64"]:
            # encode categorical variables
            X = df.copy()
            target = X[missing_column]
            X = X.drop(columns=[missing_column])
            X = encoding_categorical_variables(X)
            X[missing_column] = target
            X = X.astype(float)

            columns = list(X.columns)

            df = plugin.fit_transform(X)
            df = pd.DataFrame(df)

            # maps back the original columns
            df.columns = columns

            return df
        else:
            df = encoding_categorical_variables(df)
            df = restore_nans(df)
            df = df.astype(float)
            columns = list(df.columns)
            print("Data types after encoding: ", df.head())
            df = plugin.fit_transform(df)
            df = pd.DataFrame(df)
            df.columns = columns

            # Take the max of the "probabilities" to decide the final category, given the one hot encoding of the missing categorical variable
            col_prefix = missing_column + "_"
            targets = df.columns[df.columns.str.startswith(col_prefix)]
            print("Targets: ", targets)
            for index in df.index:
                max_value = -1
                selected_col = None
                for col in targets:
                    if df.loc[index, col] > max_value:
                        max_value = df.loc[index, col]
                        selected_col = col
                # Set all target columns to 0
                for col in targets:
                    df.loc[index, col] = 0
                # Set the selected column to 1
                if selected_col is not None:
                    df.loc[index, selected_col] = 1
            return df
        
class impute_mlp():
    def __init__(self):
        self.name = 'MLP Impute'

    def fit(self, df, missing_column):
        X = df.copy()

        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

        # Scale numerical data without pipeline
        # scaler = StandardScaler()
        # X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Pipeline for Categorical Data (One-Hot Encoding)
        # Note: Initial NaN handling is done by IterativeImputer, but OHE must be applied first.
        # categorical_cols = encoding_categorical_variables(X[categorical_cols])
        # X[categorical_cols.columns] = categorical_cols

        # if missing column is numerical
        if df[missing_column].dtype in ["int64", "float64"]:
            mlp_estimator = MLPRegressor(
                random_state=42, 
                solver='adam', 
                max_iter=100, 
                hidden_layer_sizes=(100, 50), # Example architecture
                early_stopping=True
            )

            imputer = IterativeImputer(
                estimator=mlp_estimator,
                initial_strategy='mean',
                max_iter=10,
                random_state=42
            )

            X_imputed = imputer.fit_transform(X)
            df = pd.DataFrame(X_imputed, columns=X.columns)
            return df
        else:
            mlp_estimator = MLPClassifier(
                random_state=42, 
                solver='adam', 
                max_iter=100, 
                hidden_layer_sizes=(100, 50), # Example architecture
                early_stopping=True
            )

            imputer = IterativeImputer(
                estimator=mlp_estimator,
                initial_strategy='most_frequent',
                max_iter=10,
                random_state=42
            )

            X = encoding_categorical_variables(X)
            print("Data after encoding: ", X.head())
            X_imputed = imputer.fit_transform(X)
            df = pd.DataFrame(X_imputed, columns=X.columns)
            return df


class impute_mlp_manual():
    def __init__(self):
        self.name = 'MLP Impute Manual'

    def fit(self, df: pd.DataFrame, missing_column) -> pd.DataFrame:
        X = df.copy()

        # if missing column is numerical
        if df[missing_column].dtype in ["int64", "float64"]:

            X = encoding_categorical_variables(X)

            mlp_estimator = MLPRegressor(
                random_state=42, 
                solver='adam', 
                max_iter=100, 
                hidden_layer_sizes=(100, 50), # Example architecture
                early_stopping=True
            )

            # Split the data into training and prediction sets
            train_data = X[X[missing_column].notnull()]
            predict_data = X[X[missing_column].isnull()]

            if len(predict_data) == 0:
                return df

            X_train = train_data.drop(columns=[missing_column])
            y_train = train_data[missing_column]

            X_predict = predict_data.drop(columns=[missing_column])

            # Fit the model and predict missing values
            mlp_estimator.fit(X_train, y_train)
            predicted_values = mlp_estimator.predict(X_predict)

            # Fill in the missing values
            df.loc[df[missing_column].isnull(), missing_column] = predicted_values

            return df
        
        else:

            mlp_estimator = MLPClassifier(
                random_state=42, 
                solver='adam', 
                max_iter=100, 
                hidden_layer_sizes=(100, 50), # Example architecture
                early_stopping=True
            )

            # Split the data into training and prediction sets
            train_data = X[X[missing_column].notnull()]
            predict_data = X[X[missing_column].isnull()]

            if len(predict_data) == 0:
                return df

            X_train = train_data.drop(columns=[missing_column])
            y_train = train_data[missing_column]

            X_predict = predict_data.drop(columns=[missing_column])

            # Fit the model and predict missing values
            mlp_estimator.fit(X_train, y_train)
            predicted_values = mlp_estimator.predict(X_predict)

            # Fill in the missing values
            df.loc[df[missing_column].isnull(), missing_column] = predicted_values

            return df
        
        
def main():
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

    column_to_inject_missing = columns[1]
    # inject missing values in the df, with different percentages. This data frame contains different versions of the column with missing values (different percentages)
    df_list_no_class = dirty_single_column(df[columns], column_to_inject_missing, class_name, 10)

    # This contains different versions of the dataset with missing values in the selected column
    print("Dataset before imputation: ", df_list_no_class[0].head())
    print("Dataset shape before imputation: ", df_list_no_class[0].shape)
    print("\n")

    '''
    # Lets try imputation with EM: works on numerical columns only
    imputer_em = impute_expectation_maximization()

    # for each version of the dataset with missing values, impute the missing values in the selected column
    imputer_soft = impute_soft_imputer()

    imputer_xgb = impute_xgb_imputer()

    imputer_catboost = impute_catboost()

    imputer_rfi = impute_rfi()

    imputer_autoimpute = impute_autoimpute()
    '''

    imputer_gain = impute_gain()
    imputer_vae = impute_vae()
    imputer_mlp = impute_mlp_manual()
    # Here we simulate an iteration on a list of datasets with missing values in the selected column, and we impute them one by one with
    # different techniques

    for i in range(len(df_list_no_class)):
        print("Column with missing values: ", column_to_inject_missing)
        #print(type(column_to_inject_missing))
        column_type = df_list_no_class[i][column_to_inject_missing].dtype

        print("Column type: ", column_type)
        if column_type in ["int64", "float64"]:
            print("Imputation with mlp imputer - Missing percentage: ", round(df_list_no_class[i][column_to_inject_missing].isnull().sum()/df_list_no_class[i].shape[0],2))
            df_missing = df_list_no_class[i]
            # df_missing[class_name] = df[class_name]
            df_imputed_mlp = imputer_mlp.fit(df_missing, column_to_inject_missing)
            # Check if there are still missing values
            # print("Missing values after imputation: ", df_imputed_mlp[column_to_inject_missing].isnull().sum())
            print("Imputed: ", df_imputed_mlp.head())
            print("\n")
        if column_type in ["object", "bool"]:
            print("Imputation with mlp imputer - Missing percentage: ", round(df_list_no_class[i][column_to_inject_missing].isnull().sum()/df_list_no_class[i].shape[0],2))
            df_missing = df_list_no_class[i]
            # df_missing[class_name] = df[class_name]
            df_imputed_mlp = imputer_mlp.fit(df_missing, column_to_inject_missing)
            # Check if there are still missing values
            #print("Missing values after imputation: ", df_imputed_mlp[column_to_inject_missing].isnull().sum())
            print("Imputed: ", df_imputed_mlp.head())
            print("\n")


if __name__ == "__main__":
    main()







