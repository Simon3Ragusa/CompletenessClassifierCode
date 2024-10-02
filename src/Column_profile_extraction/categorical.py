import numpy as np
import pandas as pd
import scipy as sp
import math
import time

def get_features_cat(df: pd.DataFrame, column_name: str, mask=np.ones(14, dtype=np.bool_)):
    """
    :param df: dataframe containing the data
    :param column_name: column from which we extract the features
    :return: features from the given column
    """
    # extract values and useful information
    values = df[column_name].values
    values = np.array([str(value) for value in values])
    unique_values, counts = np.unique(values[values != 'nan'], return_counts=True)
    valid_value_count = np.sum(counts)
    max_appearances = np.max(counts)
    min_appearances = np.min(counts)
    char_per_value = np.array([len(value) if value != 'nan' else np.nan for value in values])

    # extract features
    rows = values.shape[0]
    missing = round(df[column_name].isna().sum() / rows,4)
    constancy = round(max_appearances / valid_value_count,4)
    imbalance = round(max_appearances / min_appearances,4)
    uniqueness = round(unique_values.shape[0] / valid_value_count,4)

    unalikeability_sum = 0
    for i in range(len(counts)):
        for j in range(len(counts)):
            if i != j:
                unalikeability_sum += counts[i]*counts[j]
    unalikeability = round(unalikeability_sum / (valid_value_count ** 2 - valid_value_count),4)

    entr = round(entropy(df, column_name),4)
    dens = round(density(df, column_name),4)
    # per_dup = (df[column_name].duplicated().sum() / len(df[column_name])) # equal to 1 - uniqueness
    mean = round(np.nanmean(char_per_value),4)
    std = round(np.nanstd(char_per_value),4)
    minimum = round(np.nanmin(char_per_value),4)
    maximum = round(np.nanmax(char_per_value),4)
    if minimum == maximum: # there is a problem when the number of characters per value is always the same
        skew = np.nan
        kurt = np.nan
    else:
        skew = sp.stats.skew(char_per_value, nan_policy='omit')
        skew = round(1/(1+np.exp(-skew)), 4)
        kurt = sp.stats.kurtosis(char_per_value, nan_policy='omit')
        kurt = round(1/(1+math.exp(-kurt)),4)
    profile = np.array([rows, missing, constancy, imbalance, uniqueness, unalikeability, entr,
        dens, mean, std, skew, kurt, minimum, maximum])
    profile = profile[mask]

    return profile

def entropy(df, column):
    prob_attr = []
    values = df[column].values
    values = np.array([str(value) for value in values])
    for item in df[column].unique():
        if item!='nan':
            p_attr = len(df[df[column] == item]) / (len(df)-np.sum(values == 'nan'))
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
    values = np.array([str(value) for value in values])
    for item in df[column].unique():
        if item!='nan':
            p_attr = len(df[df[column] == item]) / (len(df)-np.sum(values == 'nan'))
            prob_attr.append(p_attr)
    avg_den_attr = 1 / n_distinct
    for p in prob_attr:
        den_attr += math.sqrt((p - avg_den_attr) ** 2)
        den_attr = den_attr / n_distinct
    return den_attr * 100

# This is an example
if __name__ == '__main__':
    path = "C:\\Users\\PC\\PycharmProjects\\pythonProject\\Datasets\\CSV\\"
    name = "BachChoralHarmony.csv"
    df = pd.read_csv(path + name)
    print(get_features_cat(df, column_name="V1"))