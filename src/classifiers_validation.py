import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utils import encoding_categorical_variables
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from Feature_selection.feature_selection import fixed_fs_univariate
from Imputation.imputation_techniques import impute_missing_column
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.preprocessing import OrdinalEncoder
from Column_profile_extraction.numerical import get_features_num
from Column_profile_extraction.categorical import get_features_cat
from joblib import load, dump
import itertools
import warnings
from Datasets.get_dataset import get_dataset

file_imp_methods_num = open("Imputation/methods_numerical_column.txt", "r")
file_imp_methods_cat = open("Imputation/methods_categorical_column.txt", "r")
imp_methods_num = file_imp_methods_num.readlines()
imp_methods_cat = file_imp_methods_cat.readlines()
imp_methods_num = [line.strip('\n\r') for line in imp_methods_num]
imp_methods_cat = [line.strip('\n\r') for line in imp_methods_cat]
file_imp_methods_cat.close()
file_imp_methods_num.close()


warnings.filterwarnings("ignore")

def try_classification(df, target, ml_method, baseline=False):
    """
    Computes the cross-validated f1-score of the ml method on the provided dataset.
    :param df: dataset provided for the training of the model
    :param target: name of the class column
    :param ml_method: name of the type of classifier to be trained
    :param baseline: whether we want to check the performance of the baseline
    :return:
    """
    features = list(df.columns)
    features.remove(target)
    x = df[features]
    y = df[target]
    x = encoding_categorical_variables(x)
    if baseline:
        model = DummyClassifier(strategy="most_frequent", random_state=0)
    else:
        if ml_method == "KNN":
            model = KNeighborsClassifier()
        elif ml_method == "DecisionTree":
            model = DecisionTreeClassifier(max_depth=25, random_state=0)
        elif ml_method == "LogisticRegression":
            model = LogisticRegression(C=0.35564803062231287)
        elif ml_method == "RandomForest":
            model = RandomForestClassifier(max_depth=25, n_estimators=20, random_state=0)
        else:
            model = AdaBoostClassifier()
    scaler = StandardScaler()
    pipeline = make_pipeline(scaler, model)
    cv = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    scores = cross_val_score(pipeline, x, y, scoring="f1_weighted", cv=cv)
    return np.mean(scores)

def increment_indices(indices_imp_methods):
    """
    utility function that indicate which are the next indexes of the four imputation methods
    to be tried. It is used when trying all the possible combinations of the
    four imputation methods on a validation dataset.
    :param indices_imp_methods: current indices of the imputation methods
    :return: the next indices of the imputation methods and whether to stop the computation,
    as the combinations are finished.
    """
    if indices_imp_methods[-1] < 6: # change to 7 if only numerical dataset
        indices_imp_methods[-1] += 1
    else:
        for i in range(3,-1,-1):
            if (i >= 2 and indices_imp_methods[i] == 7) or (i < 2 and indices_imp_methods[i] == 8): # change 7 to 8 for only numerical
                indices_imp_methods[i] = 0
            else:
                break
        indices_imp_methods[i] += 1

    flag = False
    for i in range(4):
        if i < 2 and indices_imp_methods[i] < 8:
            flag = True
        elif i >= 2 and indices_imp_methods[i] < 7: # change 7 to 8 for only numerical
            flag = True
    return indices_imp_methods, flag

def validate_classifiers(df, ml_method, ds_name, target, cols_to_select=4, compute=True, seed=0):
    """
    This function is used to validate the performance of the specialized classifiers
    against all possible combinations of the imputation methods. Two approaches are implemented:
    1) the suggested imputation method is applied after having imputed all
    the other columns with Standard Value Imputation. This process is repeated
    for each column, storing the imputed version of each one. The order of
    application of the suggested imputation methods does not count.
    2)  each column imputed with the suggested method is used to impute those
    that still need to be processed. In this scenario, the order of application
    of the imputation methods is relevant. The function tries all the permutations and
    saves the best and average performance.
    :param df: validation dataset
    :param ml_method: selected downstream ml method
    :param ds_name: dataset name
    :param target: name of the class column
    :param cols_to_select: number of the most important features of each type to be retained
    :param compute: whether to compute the performance of all possible combinations
    of imputation methods
    :param seed: seed for reproducibility of the results
    :return: the performance of the ml method:
    1) on the clean dataset
    2) using the first approach with the suggested imputation methods
    3) using the second approach with the suggested imputation methods
    4) with all the possible combinations of imputation methods
     (this not directly return but is saved in the Classifier_Validation folder)
    5) the list of suggested imputation methods
    """
    np.random.seed(seed)

    df_fs = fixed_fs_univariate(df, target, cols_to_select=cols_to_select)
    features = list(df_fs.columns)
    features.remove(target)
    print(features)

    # load the specialized classifiers
    num_clf = load(f"Classifier/classifiers/classifier_{ml_method}_num.joblib")
    num_scaler = load(f"Classifier/classifiers/scaler_{ml_method}_num.joblib")
    num_mask = load(f"Classifier/classifiers/features_{ml_method}_num.joblib")
    num_mask = np.array(num_mask, dtype=np.bool_)

    cat_clf = load(f"Classifier/classifiers/classifier_{ml_method}_cat.joblib")
    cat_scaler = load(f"Classifier/classifiers/scaler_{ml_method}_cat.joblib")
    cat_mask = load(f"Classifier/classifiers/features_{ml_method}_cat.joblib")
    cat_mask = np.array(cat_mask, dtype=np.bool_)

    df_dirty = df_fs[features].copy()
    perc = 0.2
    p = [perc, 1 - perc]
    for col in features:
        if col != target:
            rand = np.random.choice([True, False], size=df_dirty.shape[0], p=p)
            df_dirty.loc[rand, col] = np.nan

    # compute the suggested methods
    suggested_methods = []
    for col in features:
        column_type = df[col].dtype
        if column_type in ["int64", "float64"]:
            profile = get_features_num(df_dirty, col, num_mask)[None,:]
            profile = num_scaler.transform(profile)
            pred = num_clf.predict(profile)[0]

        else:
            profile = get_features_cat(df_dirty, col, cat_mask)
            profile = np.array([p if not np.isnan(p) else 0 for p in profile])[None,:]
            profile = cat_scaler.transform(profile)
            pred = cat_clf.predict(profile)[0]
        suggested_methods.append(pred)

    print(suggested_methods)
    # CLEAN DATASET PERFORMANCE
    f1_clean = try_classification(df_fs.copy(), target, ml_method, baseline=False)
    print("Clean dataset: ", f1_clean)

    # NO ORDER PERFORMANCE
    df_suggestions_no_order = df_dirty.copy()
    for i in range(len(features)):
        df_almost = df_dirty.copy()
        df_almost = impute_missing_column(df_almost, "impute_standard", features[i])
        df_almost[features[i]] = df_dirty[features[i]].copy()
        values = impute_missing_column(df_almost.copy(), suggested_methods[i], features[i])[features[i]]
        df_suggestions_no_order[features[i]] = values

    df_suggestions_no_order[target] = df_fs[target]
    f1_sugg_no_order = try_classification(df_suggestions_no_order, target, ml_method, False)
    print("With suggestions (no order): ", f1_sugg_no_order)

    # ALL PERMUTATIONS OF SUGGESTED METHODS PERFORMANCE
    order_performance_list = []
    permutations = list(itertools.permutations(range(4),4))
    for ordering in permutations:
        print(ordering)
        df_suggestions_order = df_dirty.copy()
        df_suggestions_order = impute_missing_column(df_suggestions_order,"impute_standard", None)
        for i in ordering:
            df_suggestions_order[features[i]] = df_dirty[features[i]].copy()
            values = impute_missing_column(df_suggestions_order.copy(), suggested_methods[i], features[i])[features[i]]
            df_suggestions_order[features[i]] = values

        df_suggestions_order[target] = df_fs[target]
        f1_sugg_order = try_classification(df_suggestions_order, target, ml_method, False)
        order_performance_list.append(f1_sugg_order)


    f1_sugg_order = np.max(order_performance_list)
    f1_sugg_min = np.min(order_performance_list)
    f1_sugg_index = np.argmax(order_performance_list)
    f1_sugg_worst_index = np.argmin(order_performance_list)
    f1_sugg_mean = np.mean(order_performance_list)
    best_order = permutations[f1_sugg_index]
    worst_order = permutations[f1_sugg_worst_index]

    # print(np.array(permutations)[np.argsort(order_performance_list)[::-1]])
    # print(np.sort(order_performance_list)[::-1])
    print("BEST SUGGESTED ORDER: \n", best_order, f1_sugg_order)
    print("WORST SUGGESTED ORDER: \n", worst_order, f1_sugg_min)
    # print(suggested_methods)
    print("With suggestions (with order): ", f1_sugg_order)

    if not compute: # stop here if all combinations have been previously computed
        return f1_clean, f1_sugg_no_order, f1_sugg_order, f1_sugg_mean, suggested_methods

    # ALL COMBINATIONS PERFORMANCE

    col_dict = {col: [] for col in features}

    for i, col in enumerate(features):
        print(f"Starting {col}")
        col_type = df[col].dtype
        df_almost = df_dirty.copy()
        df_almost = impute_missing_column(df_almost, "impute_standard", col)
        df_almost[col] = df_dirty[col].copy()
        if col_type in ["int64", "float64", "int32"]:
            for imp_method in imp_methods_num:
                values = impute_missing_column(df_almost.copy(), imp_method, col)[col]
                col_dict[col].append(values)
        else:
            for imp_method in imp_methods_cat:
                values = impute_missing_column(df_almost.copy(), imp_method, col)[col]
                col_dict[col].append(values)
        print(f"Done {col}")

    indices_imp_methods = np.zeros(len(features), dtype=np.int64)
    f1_comb_list = []
    curr_iter = 1
    flag = True
    seen_greater = 0
    while flag:
        curr_iter += 1
        print(f"{curr_iter}")
        df_dirty_comb = pd.DataFrame(columns=features)
        for i,col in enumerate(features):
            df_dirty_comb[col] = col_dict[col][indices_imp_methods[i]]

        df_dirty_comb[target] = df_fs[target]
        f1_comb = try_classification(df_dirty_comb, target, ml_method, False)
        f1_comb_list.append(f1_comb)
        if f1_comb > f1_sugg_no_order:
            seen_greater += 1
            print(f"ALERT {seen_greater} / {f1_comb}")
        indices_imp_methods, flag = increment_indices(indices_imp_methods)
    dump(f1_comb_list, f"Classifier_Validation/{ds_name}/list_f1_scores_{ml_method}.joblib")
    return f1_clean, f1_sugg_no_order, f1_sugg_order, f1_sugg_mean, suggested_methods

def analyze_list(f1_clean, f1_sugg_no_order, f1_sugg_order, f1_sugg_mean, ml_method, ds_name):
    f1_list = load(f"Classifier_Validation/{ds_name}/list_f1_scores_{ml_method}.joblib")
    f1_list = np.array(f1_list)

    quant_25 = np.quantile(f1_list, 0.25)
    quant_75 = np.quantile(f1_list, 0.75)
    median = np.median(f1_list)

    plt.figure()
    plt.hist(f1_list, bins=25)
    plt.axvline(x=f1_clean, c="brown", label="clean")
    plt.axvline(x=f1_sugg_no_order, c="red", label="suggested (no order)")
    plt.axvline(x=f1_sugg_order, c="lime", label="suggested (order)")
    plt.axvline(x=quant_25, c="purple", label="25 percentile", ls="--")
    plt.axvline(x=quant_75, c="orange", label="75 percentile", ls="--")
    plt.axvline(x=median, c="black", label="median", ls="--")
    plt.legend()
    plt.title(f"Dataset '{ds_name}': {ml_method}")
    plt.xlabel("f1 score")
    plt.ylabel("number of combinations")
    plt.savefig(f"Classifier_Validation/{ds_name}/{ml_method}")
    plt.show()

    print(round(f1_clean, 4), round(quant_25, 4), round(median, 4),  round(quant_75, 4),
          round(f1_sugg_no_order, 4),
          round(f1_sugg_order, 4),
          round(f1_sugg_mean,4))


if __name__ == "__main__":
    ml_methods = ["DecisionTree", "LogisticRegression","KNN","RandomForest","AdaBoost"]
    name_main = "consumer" # ["wine", "visualizing_galaxy", "consumer", "student"]
    target_main = "PurchaseIntent" # ["Wine", "binaryClass", "PurchaseIntent", "GradeClass"]
    df_main = get_dataset("Datasets/CSV/", f"{name_main}.csv")

    # df_main.drop("StudentID", inplace=True, axis=1) # only used with dataset student

    df = df_main.copy()
    selector = SelectKBest(mutual_info_classif, k='all')
    features = list(df_main.columns)
    features.remove(target_main)
    cat_cols = list(
        df_main.select_dtypes(include=['bool', 'object']).columns)
    df[cat_cols] = OrdinalEncoder().fit_transform(df[cat_cols])
    selector.fit(df[features], df[target_main])

    # print the most important features, in order
    print(np.array(selector.get_feature_names_out())[np.argsort(selector.scores_)][::-1])

    if name_main == "consumer" or name_main == "student":
        cols_to_select = 2

    else: # visualizing_galaxy or wine are fully numerical
        cols_to_select = 4

    for ml_method_main in ml_methods:
        print(ml_method_main)

        # in the following, change the compute parameter if the performance of all combinations has been already computed
        f1_clean_main, f1_sugg_no_order_main, f1_sugg_order_main, f1_sugg_mean_main, suggested_methods = validate_classifiers(df_main,
                                                                                                                              ml_method_main,
                                                                                                                              name_main,
                                                                                                                              target_main,
                                                                                                                              cols_to_select=cols_to_select,
                                                                                                                              compute=True)

        analyze_list(f1_clean_main, f1_sugg_no_order_main, f1_sugg_order_main, f1_sugg_mean_main, ml_method_main, name_main)
