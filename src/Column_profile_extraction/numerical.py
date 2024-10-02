import numpy as np
import pandas as pd
import scipy as sp
import math


def get_features_num(df, column_name, mask=np.ones(20, dtype=np.bool_)):
    """
    :param df: dataframe containing the data
    :param column_name: column from which we extract the features
    :return: features from the given column
    """
    values = df[column_name].copy().values
    rows = values.shape[0]
    missing = round(np.sum(np.isnan(values)) / rows, 4)
    uniqueness = round(np.unique(values[~np.isnan(values)]).shape[0] / (rows-np.sum(np.isnan(values))), 4)
    minimum = round(np.nanmin(values), 4)
    maximum = round(np.nanmax(values), 4)
    mean = round(np.nanmean(values), 4)
    median = round(np.nanmedian(values), 4)
    std = round(np.nanstd(values), 4)
    skew = round(sp.stats.skew(values, nan_policy='omit'),4)
    kurt = round(sp.stats.kurtosis(values, nan_policy='omit'), 4)
    mad = round(np.nanmedian(np.abs(values - median)), 4)
    iqr = round(np.nanquantile(values, 0.75) - np.nanquantile(values, 0.25), 4)
    p_min, p_max = correlations2(df, column_name, "pearson")
    s_min, s_max = correlations2(df, column_name, "spearman")
    k_min, k_max = correlations2(df, column_name, "kendall")
    entr = round(entropy(df, column_name), 4)
    dens = round(density(df, column_name), 4)
    profile = np.array([rows, missing, uniqueness, minimum, maximum, mean, median, std, skew, kurt, mad, iqr, p_min, p_max, k_min, k_max, s_min, s_max, entr, dens])
    profile = profile[mask]
    return profile


# This function is probably inefficient
def correlations(df: pd.DataFrame, column_name: str, method: str = "pearson"):
    num = list(df.select_dtypes(include=['int64', 'float64',"int32"]).columns)
    if len(num) > 1:
        corr_matrix = df[num].corr(method=method)
        corr_matrix.drop(column_name, axis=0,
                         inplace=True)  # avoid to select correlation with itself
        return (round(corr_matrix[column_name].abs().min(),4),
                round(corr_matrix[column_name].abs().max(),4))
    else:
        return 0., 0.


def correlations2(df: pd.DataFrame, column_name: str, method: str = "pearson"):
    num = list(df.select_dtypes(include=['int64', 'float64',"int32"]).columns)
    if len(num) > 1:
        correlations = df[num].corrwith(df[column_name], method=method).values
        correlations = np.delete(correlations, df[num].columns.get_loc(column_name))
        correlations = np.abs(correlations)
        return round(np.min(correlations),4), round(np.max(correlations),4)
    else:
        return 0., 0.

def entropy(df, column):
    prob_attr = []
    values = df[column].values
    for item in df[column].unique():
        if not np.isnan(item):
            p_attr = len(df[df[column] == item]) / (len(df)-np.sum(np.isnan(values)))
            prob_attr.append(p_attr)
    en_attr = 0
    if 0 in prob_attr:
        prob_attr.remove(0)
    for p in prob_attr:
        en_attr += p * np.log(p)
    en_attr = -en_attr
    return en_attr

def density(df, column):
    n_distinct = df[column].nunique()
    prob_attr = []
    den_attr = 0
    values = df[column].values
    for item in df[column].unique():
        if not np.isnan(item):
            p_attr = len(df[df[column] == item])/(len(df)-np.sum(np.isnan(values)))
            prob_attr.append(p_attr)
    avg_den_attr = 1/n_distinct
    for p in prob_attr:
        den_attr += math.sqrt((p - avg_den_attr) ** 2)
        den_attr = den_attr/n_distinct
    return den_attr*100


# this is an example
if __name__ == '__main__':
    path = "C:\\Users\\PC\\PycharmProjects\\pythonProject\\Datasets\\CSV\\"
    name = "iris.csv"
    df = pd.read_csv(path + name)
    print(get_features_num(df, column_name="petal_length"))

