import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    features_num = pd.read_csv(
        "features_numerical_columns.csv")
    features = list(features_num.columns)
    features.remove("name")
    features.remove("column_name")

    X = features_num[features]
    print(features_num.shape)
    print(len(features))

    plt.figure(figsize=(18,14))
    sns.heatmap(X.corr(), annot = True, cmap = 'vlag_r', vmin = -1, vmax = 1)
    print(X.columns)
    plt.show()


    features_cat = pd.read_csv(
        "features_categorical_columns.csv")
    features = list(features_cat.columns)
    features.remove("name")
    features.remove("column_name")

    X = features_cat[features]
    plt.figure(figsize=(18, 14))
    sns.heatmap(X.corr(), annot=True, cmap='vlag_r', vmin=-1,
                vmax=1)
    print(X.columns)
    plt.show()

