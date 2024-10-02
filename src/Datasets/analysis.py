import pandas as pd
import numpy as np
from get_dataset import get_dataset
from src.Feature_selection.feature_selection import feature_selection_univariate
import matplotlib.pyplot as plt
import seaborn as sns


def write():
    path_datasets = "CSV/"
    file = open("analysis.csv", "w")
    file_datasets = open("dataset_names.txt", "r")
    file.write("name,column_name," +
               "min,max,mean,median,std,mad," +
               "iqr\n")
    datasets = file_datasets.readlines()
    datasets = [line.strip('\n\r') for line in datasets]
    for dataset in datasets:
        print("----------" + dataset + "----------")
        df = get_dataset(path_datasets,dataset + ".csv")
        class_name = df.columns[-1]
        df, _, _, _, _ = feature_selection_univariate(df, class_name,
                                                      perc_num=50, perc_cat=60)
        columns = list(df.columns)
        columns.remove(class_name)
        current_dataset_features = []
        for column in columns:
            print("analyzing column: ", column)
            if df[column].dtype == "int64" or df[column].dtype == "float64":
                values = df[column].values
                features = get_features_here(values)
                current_dataset_features.append(features)
                new_line = str(dataset) + "," + str(column) + ","
                for feature in features:
                    new_line += str(feature) + ","
                new_line = new_line[:-1]
                new_line += "\n"
                file.write(new_line)
    file.close()
    file_datasets.close()

def get_features_here(vals):
    min = np.min(vals)
    max = np.max(vals)
    mean = np.mean(vals)
    median = np.median(vals)
    std = np.std(vals)
    mad = np.nanmedian(np.abs(vals - median))
    iqr = np.nanquantile(vals, 0.75) - np.nanquantile(vals, 0.25)
    return min,max,mean,median,std,mad,iqr


def corr():
    df = pd.read_csv("analysis.csv")
    features = list(df.columns)
    features.remove("name")
    features.remove("column_name")

    X = df[features]

    plt.figure(figsize=(18, 14))
    sns.heatmap(X.corr(), annot=True, cmap='vlag_r', vmin=-1, vmax=1)
    plt.show()
    maxs = X["max"].values
    means = X["std"].values

if __name__ == "__main__":
    write()
    corr()

