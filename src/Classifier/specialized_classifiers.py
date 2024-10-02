import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import random
import shap
from joblib import dump

ml_methods = ["DecisionTree", "LogisticRegression", "KNN", "RandomForest", "AdaBoost"]
file_imp_methods_num = open("../Imputation/methods_numerical_column.txt", "r")
file_imp_methods_cat = open("../Imputation/methods_categorical_column.txt", "r")
imp_methods_num = file_imp_methods_num.readlines()
imp_methods_cat = file_imp_methods_cat.readlines()
imp_methods_num = [line.strip('\n\r') for line in imp_methods_num]
imp_methods_cat = [line.strip('\n\r') for line in imp_methods_cat]
np.random.seed(42)

def convert_to_list(value):
    return ast.literal_eval(value)


def filter_equivalency(df, max_imp_methods=4):
    """
    This function filters out the units of knowledge (rows) associated with too many
    (expressed by max_imp_methods parameter) equivalent imputation methods. It is used for
    building the training dataset for the specialized classifiers, that must learn from units
    of knowledge that expresses a distinct preference towards some imputation methods.
    :param df: the Dataframe from which we want to filter the rows having at most
    max_imp_methods equivalent imputation methods
    :param max_imp_methods: the maximum allowed number of equivalent imputation methods
    allowed for being retained in the dataframe
    :return: the filtered dataframe
    """
    df_filtered = pd.DataFrame(columns=df.columns)
    all_imp = []
    row_filtered = 0
    best_methods = df["best_methods"]
    for i in df.index:
        methods = best_methods[i] # list of imputation methods for current row
        if len(methods) <= max_imp_methods:
            df_filtered.loc[row_filtered, :] = df.loc[i,:]
            row_filtered += 1
            all_imp.append(methods[0])
    df_filtered = df_filtered.convert_dtypes()
    return df_filtered

def bar_plots(is_num=True):
    """
    Saves in a folder a barplot for each downstream ml method. The barplot counts
    how many times an imputation method will be used as label for building the
    specialized classifier for that ml method.
    :param is_num: True for analyzing the numerical columns, false for the categorical
    columns
    """
    for ml_method in ml_methods:
        if is_num:
            df = pd.read_csv(
                "../Experiments/combined_new_prova/numerical_kb_combined.csv", converters={'best_methods': convert_to_list})
            df = df[df["ml_algorithm"] == ml_method]
            path = "C:/Users/PC/Desktop/risultati/counts/num/"
        else:
            df = pd.read_csv(
                "../Experiments/combined_new_prova/categorical_kb_combined.csv", converters={'best_methods': convert_to_list})
            df = df[df["ml_method"] == ml_method]
            path = "C:/Users/PC/Desktop/risultati/counts/cat/"
        df = filter_equivalency(df, max_imp_methods=4)
        best_methods = df["best_methods"]
        first_methods = []
        for i in range(len(best_methods)):
            first_methods.append(best_methods[i][0])

        vals, counts = np.unique(first_methods, return_counts=True)

        plt.figure(figsize=(10,8))
        plt.title(f"{ml_method}. Number of samples: {np.sum(counts)}")
        plt.bar(vals, counts)
        plt.xticks(rotation=20)
        plt.savefig(path + f"{ml_method}.png")
        plt.show()


def classifier(clf, is_num=True, ml_method="DecisionTree", baseline_random=False, baseline_zero_r=False, is_shap=False):
    """
    :param clf: sklearn classifier to be trained and validated on a dataset
    :param is_num: whether train on numerical or categorical units of knowledge. Expects
    a boolean value
    :param ml_method: selected downstream ml method, must be among the allowed ones
    :param baseline_random: whether print metrics for the random guess baseline
    :param baseline_zero_r: whether print metrics for the zeroR baseline
    :param is_shap: wheter compute shap values on the validation folds for the classifier
    :return: the cross-validated accuracy of the classifier
    """

    # retrieve from the files the units of knowledge for the ml method
    if is_num:
        df = pd.read_csv(
            "../Experiments/combined_new_prova/numerical_kb_combined.csv", converters={'best_methods': convert_to_list})
        df = df[df["ml_algorithm"] == ml_method]
        plot_dir = "C:/Users/PC/Desktop/risultati/shap/num/"
    else:
        df = pd.read_csv(
            "../Experiments/combined_new_prova/categorical_kb_combined.csv", converters={'best_methods': convert_to_list})
        df = df[df["ml_method"] == ml_method]
        plot_dir = "C:/Users/PC/Desktop/risultati/shap/cat/"

    # filter out units of knowledge with too many equivalent imputation methods
    df = filter_equivalency(df, max_imp_methods=4)

    # read the training dataset names (for cross-validation)
    file_datasets = open("../Datasets/dataset_names.txt", "r")
    datasets = file_datasets.readlines()
    datasets = [line.strip('\n\r') for line in datasets]
    file_datasets.close()

    # filter out too correlated features, otherwise shap values might be misleading
    threshold = 0.8
    corr = df.corr(numeric_only=True)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df.drop(df[to_drop], axis=1, inplace=True)

    # fill the possible nan values
    df = df.fillna(0)

    # define class and features of the problem
    class_name = "best_methods"
    feature_cols = list(df.columns)
    feature_cols.remove("name")
    feature_cols.remove("column_name")
    feature_cols.remove("best_methods")
    if is_num:
        feature_cols.remove("ml_algorithm")
    else:
        feature_cols.remove("ml_method")

    # here starts the cross_validation
    seen_samples = 0
    accuracies = []
    counts = []
    counts_list = []

    # avoid errors caused by different number of classes for different ml methods
    if is_num:
        class_names = imp_methods_num.copy()
        if ml_method == "RandomForest":
            class_names.remove("impute_mice")

        if ml_method != "RandomForest":
            all_shap_values = np.array([]).reshape((len(imp_methods_num),0,len(feature_cols)))
        else:
            all_shap_values = np.array([]).reshape((len(imp_methods_num)-1, 0, len(feature_cols)))
    else:

        class_names = imp_methods_cat.copy()
        if ml_method == "LogisticRegression":
            class_names.remove("impute_mode")
            class_names.remove("impute_knn")

        if ml_method == "LogisticRegression":
            all_shap_values = np.array([]).reshape((len(imp_methods_cat)-2,0,len(feature_cols)))
        else:
            all_shap_values = np.array([]).reshape((len(imp_methods_cat), 0, len(feature_cols)))

    class_names.sort()
    idxs = []
    for dataset in datasets:
        # columns of different datasets could have the same name
        columns_names = df[df["name"] == dataset]["column_name"].unique()
        for column_name in columns_names:

            # define training and test set
            training_set = df[(df["name"] != dataset) | (df["column_name"] != column_name)]
            test_set = df[(df["name"] == dataset) & (df["column_name"] == column_name)]

            x_train = training_set[0:][feature_cols]  # Features of training
            best_methods_train = training_set[0:][class_name].values
            y_train = [] # target
            for i in range(len(best_methods_train)):
                y_train.append(best_methods_train[i][0])

            x_test = test_set[0:][feature_cols]  # Features of test

            # scaling the data
            scaler = RobustScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            best_methods_test = test_set[0:][class_name].values

            # baseline training and prediction
            if baseline_random:
                vals, apps = np.unique(y_train, return_counts=True)
                probs = apps/np.sum(apps)
                y_pred = np.random.choice(vals, size=x_test.shape[0], p=probs)
            elif baseline_zero_r:
                vals, apps = np.unique(y_train, return_counts=True)
                most_frequent = vals[np.argmax(apps)]
                y_pred = np.array([most_frequent for _ in range(x_test.shape[0])])

            # classifier training
            else:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

                # shap computation on the test set
                if is_shap:
                    if isinstance(clf, RandomForestClassifier):
                        explainer = shap.TreeExplainer(clf, seed=0, output_names=clf.classes_)
                        shap_values = explainer.shap_values(x_test)
                    else: # clf is an SVC
                        explainer = shap.KernelExplainer(clf.predict_proba, data=shap.sample(x_train, 100), seed=0, output_names=clf.classes_)
                        shap_values = explainer.shap_values(x_test)

                    # avoid errors for different folds having different numbers of classes
                    if is_num:
                        if ml_method != "RandomForest":
                            if len(shap_values) == len(imp_methods_num):
                                all_shap_values = np.concatenate([all_shap_values, shap_values], axis=1)
                            else:
                                for i in test_set.index:
                                    idxs.append(i)
                        else:
                            all_shap_values = np.concatenate([all_shap_values, shap_values], axis=1)

                    else:
                        if ml_method != "LogisticRegression":
                            if len(shap_values) == len(imp_methods_cat):
                                all_shap_values = np.concatenate([all_shap_values, shap_values], axis=1)
                            else:
                                for i in test_set.index:
                                    idxs.append(i)
                        elif ml_method == "LogisticRegression":
                            if len(shap_values) == 6:
                                all_shap_values = np.concatenate([all_shap_values, shap_values], axis=1)
                            else:
                                for i in test_set.index:
                                    idxs.append(i)
            # compute accuracy on the fold
            accuracy = 0
            for i, y in enumerate(y_pred):
                counts_list.append(len(best_methods_test[i]))
                if y in best_methods_test[i]:
                    accuracy += 1

            accuracy /= y_pred.shape[0]
            accuracies.append(accuracy)
            counts.append(y_pred.shape[0])
            seen_samples += y_pred.shape[0]
    # compute mean accuracy
    accuracies = np.array(accuracies)
    counts = np.array(counts)
    mean_accuracy = np.sum(accuracies*counts)/seen_samples

    # compute and save shap values
    if is_shap:
        type_col = "categorical" if not is_num else "numerical"
        df_shap = df.copy()
        if len(idxs) != 0:
            df_shap = df.drop(idxs)
        df_shap = df_shap[feature_cols]
        shap_as_list = shap_values_to_list(all_shap_values, ml_method, is_num)
        fig = plt.figure()
        plt.title(ml_method + ": " + type_col + " columns")
        shap.summary_plot(shap_as_list, df_shap, plot_type="bar", class_names=clf.classes_, max_display=3, plot_size=[8,4])
        fig.savefig(plot_dir + f"{ml_method}_{type_col}.pdf")

        for i in range(len(class_names)):
            fig = plt.figure()
            plt.title(f"{ml_method}: {clf.classes_[i]}")
            shap.summary_plot(shap_as_list[i], df_shap, class_names=clf.classes_)
            fig.savefig(plot_dir + "swarm_plots/" + f"{ml_method}/" + f"{ml_method}_{clf.classes_[i]}.png")

    return mean_accuracy


def shap_values_to_list(shap_values, ml_method, is_num=True):
    """
    utility function for shap values computation
    """
    shap_as_list = []
    if is_num:
        if ml_method != "RandomForest":
            tot = len(imp_methods_num)
        else:
            tot = len(imp_methods_num) - 1
    else:
        if ml_method != "LogisticRegression":
            tot = len(imp_methods_cat)
        else:
            tot = 6
    for i in range(tot):
        shap_as_list.append(shap_values[i,:,:])
    return shap_as_list

def tune_rf(max_iter=100, is_num=True):
    """
    Tries random initializations of a Random Forest classifier and returns the best
    cross-validated set of hyperparameters for all ml methods.
    :param max_iter: maximum number of to-be-tried hyperparameter initializations
    :param is_num: wether fit on numerical units of knowledge or categorical ones
    """
    n_estimators = [int(x) for x in np.linspace(start=5, stop=200, num=40)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 200, num=20)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10, 15]
    min_samples_leaf = [1, 2, 4, 8]
    criterion = ["gini", "log_loss", "entropy"]
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion': criterion}
    max_acc = np.zeros(len(ml_methods))
    best_params = {key: [] for key in range(len(ml_methods))}
    for curr_iter in range(max_iter):
        print(curr_iter + 1)
        params = []
        for key in random_grid:
            params.append(random.choice(random_grid[key]))
        print(params)
        for i, ml_method in enumerate(ml_methods):
            clf = RandomForestClassifier(n_estimators=params[0],
                                         max_features=params[1],
                                         max_depth=params[2],
                                         min_samples_split=params[3],
                                         min_samples_leaf=params[4],
                                         bootstrap=params[5],
                                         random_state=0)

            acc = classifier(clf, is_num=is_num, ml_method=ml_method, is_shap=False)
            # print(ml_method + ": " + str(acc))
            if max_acc[i] < acc:
                max_acc[i] = acc
                print(f"I changed params for {ml_method}, new accuracy is: " + str(round(acc,4)))
                best_params[i] = params.copy()

    for i in range(max_acc.shape[0]):
        print(f"ALGORITHM {i}: ")
        print(max_acc[i])
        print(best_params[i])

def tune_svc(max_iter=100, is_num=True):
    """
    Tries random initializations of a Support Vector classifier and returns the best
    cross-validated set of hyperparameters for all ml methods.
    :param max_iter: maximum number of to-be-tried hyperparameter initializations
    :param is_num: wether fit on numerical units of knowledge or categorical ones
    """
    random_grid = {'C': np.linspace(0.001, 20, 10000),
                   'gamma': [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 'auto', 'scale'],
                   'kernel': ['rbf']}
    max_acc = np.zeros(len(ml_methods))
    best_params = {key: [] for key in range(len(ml_methods))}
    for curr_iter in range(max_iter):
        print(curr_iter + 1)
        params = []
        for key in random_grid:
            params.append(random.choice(random_grid[key]))
        print(params)
        for i, ml_method in enumerate(ml_methods):
            clf = SVC(C=params[0],
                      gamma=params[1],
                      kernel=params[2])

            acc = classifier(clf, is_num=is_num, ml_method=ml_method)
            # print(ml_method + ": " + str(acc))
            if max_acc[i] < acc:
                max_acc[i] = acc
                print(
                    f"I changed params for {ml_method}, new accuracy is: " + str(
                        round(acc, 4)))
                best_params[i] = params.copy()

    for i in range(max_acc.shape[0]):
        print(f"ALGORITHM {i}: ")
        print(max_acc[i])
        print(best_params[i])

def get_model(ml_method, is_num):
    """
    This function returns the model having the best cross-validated performance
    after hyperparameter tuning
    :param ml_method: the selected downstream classification method
    :param is_num: whether return the model for categorical or numerical columns
    :return:
    """
    # parameters of the models are retrieved from the tuning (tune_rf and tune_svc)
    if is_num:
        if ml_method == "DecisionTree":
            params = [55, 'log2', 190, 10, 4, False, 'gini']
            clf = RandomForestClassifier(n_estimators=params[0],
                                         max_features=params[1],
                                         max_depth=params[2],
                                         min_samples_split=params[3],
                                         min_samples_leaf=params[4],
                                         bootstrap=params[5],
                                         random_state=0)

        elif ml_method == "LogisticRegression":
            params = [7.849392439243924, 0.5, 'rbf']
            clf = SVC(C=params[0],
                      gamma=params[1],
                      kernel=params[2],
                      probability=True,
                      random_state=0)

        elif ml_method == "KNN":
            params = [125, 'log2', 130, 5, 2, True, 'log_loss']
            clf = RandomForestClassifier(n_estimators=params[0],
                                         max_features=params[1],
                                         max_depth=params[2],
                                         min_samples_split=params[3],
                                         min_samples_leaf=params[4],
                                         bootstrap=params[5],
                                         random_state=0)

        elif ml_method == "RandomForest":
            params = [12.027601360136012, 0.001, 'rbf']
            clf = SVC(C=params[0],
                      gamma=params[1],
                      kernel=params[2],
                      probability=True,
                      random_state=0)

        else:
            params = [190, 'sqrt', 150, 10, 8, True, 'log_loss']
            clf = RandomForestClassifier(n_estimators=params[0],
                                         max_features=params[1],
                                         max_depth=params[2],
                                         min_samples_split=params[3],
                                         min_samples_leaf=params[4],
                                         bootstrap=params[5],
                                         random_state=0)

    else:
        if ml_method == "DecisionTree":
            params = [10, 'sqrt', 40, 10, 8, True, 'gini']
            clf = RandomForestClassifier(n_estimators=params[0],
                                         max_features=params[1],
                                         max_depth=params[2],
                                         min_samples_split=params[3],
                                         min_samples_leaf=params[4],
                                         bootstrap=params[5],
                                         random_state=0)

        if ml_method == "LogisticRegression":
            params = [3.4531726172617256, 0.01, 'rbf']
            clf = SVC(C=params[0],
                      gamma=params[1],
                      kernel=params[2],
                      probability=True,
                      random_state=0)

        elif ml_method == "KNN":
            params = [16.027801380138012, 'scale', 'rbf']
            clf = SVC(C=params[0],
                      gamma=params[1],
                      kernel=params[2],
                      probability=True,
                      random_state=0)

        elif ml_method == "RandomForest":
            params = [15.62378117811781, 0.001, 'rbf']
            clf = SVC(C=params[0],
                      gamma=params[1],
                      kernel=params[2],
                      probability=True,
                      random_state=0)

        else:
            params = [0.013000600060006001, 0.5, 'poly', 3]
            clf = SVC(C=params[0],
                      gamma=params[1],
                      kernel=params[2],
                      degree=params[3],
                      probability=True,
                      random_state=0)

    return clf

def inspect_shap(is_num):
    """
    function that saves the shap values for each ml method
    :param is_num: whether work on numerical or categorical columns
    :return:
    """
    for ml_method in ml_methods:
        clf = get_model(ml_method, is_num)
        classifier(clf, is_num, ml_method, is_shap=True)


def train_classifiers_whole_dataset_num():
    """
    This functions train and saves the best models for numerical columns,
    to be used for their validation on untrained dataset
    """
    for ml_method in ml_methods:
        clf = get_model(ml_method, True)
        df = pd.read_csv(
            "../Experiments/combined_new_prova/numerical_kb_combined.csv", converters={'best_methods': convert_to_list})
        df = df[df["ml_algorithm"] == ml_method]
        df = filter_equivalency(df, max_imp_methods=4)

        mask = np.zeros(len(list(df.columns)) - 4)
        full_profile = list(df.columns)
        full_profile.remove("name")
        full_profile.remove("column_name")
        full_profile.remove("best_methods")
        full_profile.remove("ml_algorithm")

        threshold = 0.8
        corr = df.corr(numeric_only=True)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if
                   any(upper[column] > threshold)]
        df.drop(df[to_drop], axis=1, inplace=True)

        df = df.fillna(0)

        class_name = "best_methods"
        feature_cols = list(df.columns)
        feature_cols.remove("name")
        feature_cols.remove("column_name")
        feature_cols.remove("best_methods")
        feature_cols.remove("ml_algorithm")

        for i in range(len(full_profile)):
            if full_profile[i] in feature_cols:
                mask[i] = 1

        x = df[feature_cols]
        best_methods_train = df[class_name]

        y = []
        for i in range(len(best_methods_train)):
            y.append(best_methods_train[i][0])

        scaler = RobustScaler()
        x = scaler.fit_transform(x)

        clf.fit(x, y)

        dump(clf, f'classifiers/classifier_{ml_method}_num.joblib')
        dump(scaler, f'classifiers/scaler_{ml_method}_num.joblib')
        dump(mask, filename=f"classifiers/features_{ml_method}_num.joblib")
        print(f"done {ml_method} num")


def train_classifiers_whole_dataset_cat():
    """
    This functions train and saves the best models for categorical columns,
    to be used for their validation on untrained dataset
    """
    for ml_method in ml_methods:
        clf = get_model(ml_method, False)
        df = pd.read_csv(
            "../Experiments/combined_new_prova/categorical_kb_combined.csv", converters={'best_methods': convert_to_list})
        df = df[df["ml_method"] == ml_method]
        df = filter_equivalency(df, max_imp_methods=4)

        mask = np.zeros(len(list(df.columns))-4)
        full_profile = list(df.columns)
        full_profile.remove("name")
        full_profile.remove("column_name")
        full_profile.remove("best_methods")
        full_profile.remove("ml_method")

        threshold = 0.8
        corr = df.corr(numeric_only=True)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        df.drop(df[to_drop], axis=1, inplace=True)

        df = df.fillna(0)

        class_name = "best_methods"
        feature_cols = list(df.columns)
        feature_cols.remove("name")
        feature_cols.remove("column_name")
        feature_cols.remove("best_methods")
        feature_cols.remove("ml_method")

        for i in range(len(full_profile)):
            if full_profile[i] in feature_cols:
                mask[i] = 1

        x = df[feature_cols]
        best_methods_train = df[class_name]

        y = []
        for i in range(len(best_methods_train)):
            y.append(best_methods_train[i][0])

        scaler = RobustScaler()
        x = scaler.fit_transform(x)

        clf.fit(x, y)

        dump(clf, f'classifiers/classifier_{ml_method}_cat.joblib')
        dump(scaler, f'classifiers/scaler_{ml_method}_cat.joblib')
        dump(mask, filename=f"classifiers/features_{ml_method}_cat.joblib")
        print(f"done {ml_method} cat")


if __name__ == "__main__":

    # pipeline for the training and inspection of the specialized classifiers
    tune_rf(is_num=True)
    tune_rf(is_num=False)
    tune_svc(is_num=True)
    tune_svc(is_num=False)

    # baseline computation
    for ml_method_main in ml_methods:
        num_list = []
        cat_list = []
        for _ in range(10):
            num_list.append(classifier(clf=0, baseline_zero_r=True, is_num=True, ml_method=ml_method_main))
            cat_list.append(classifier(clf=0, baseline_zero_r=True, is_num=False, ml_method=ml_method_main))
        print(ml_method_main)
        print("num baseline: ", np.mean(num_list))
        print("cat baseline: ", np.mean(cat_list))

    # feature importance inspection
    inspect_shap(is_num=True)
    inspect_shap(is_num=False)
    bar_plots(is_num=True)
    bar_plots(is_num=False)

    train_classifiers_whole_dataset_num()
    train_classifiers_whole_dataset_cat()
