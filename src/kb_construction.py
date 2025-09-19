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

# opening files with names of datasets, ml algorithms and imputation methods
file_datasets = open("src/Datasets/dataset_names.txt", "r")
file_ml_methods = open("src/Classification/classification_methods.txt", "r")
file_imp_methods_num = open("src/Imputation/methods_numerical_column.txt", "r")
file_imp_methods_cat = open("src/Imputation/methods_categorical_column.txt", "r")

datasets = file_datasets.readlines()
ml_methods = file_ml_methods.readlines()
imp_methods_num = file_imp_methods_num.readlines()
imp_methods_cat = file_imp_methods_cat.readlines()

datasets = [line.strip('\n\r') for line in datasets]
ml_methods = [line.strip('\n\r') for line in ml_methods]
imp_methods_num = [line.strip('\n\r') for line in imp_methods_num]
imp_methods_cat = [line.strip('\n\r') for line in imp_methods_cat]

# this dataframe contains the value of the parameters to train the ml algorithms
df_hyper = pd.read_csv("src/Hyperparameter_tuning/hyperparameters.csv")

def generate_seed(n_seed, n_elements):
    seed = []
    seeds = []
    for r in range(0, n_seed):
        for i in range(0, n_elements):
            seed.append(int(np.random.randint(0, 100)))
        seeds.append(seed)
        seed = []
    return seeds


def parallel_exec(df, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds):
    n_instances_x_job = int(n_instances_tot / n_parallel_jobs)
    seed = generate_seed(n_parallel_jobs, n_instances_x_job)

    # write the seeds in the seeds file
    flat_seeds = [x[0] for x in seed]
    new_line_seeds = dataset + "," + column + ","
    for s in flat_seeds:
        new_line_seeds += str(s) + ","
    new_line_seeds = new_line_seeds[:-1] + "\n"
    file_seeds.write(new_line_seeds)

    itr = zip(repeat(df), repeat(dataset), repeat(class_name), repeat(column), seed)

    # starts the parallel experiments on the column
    with (Pool(processes=n_parallel_jobs) as pool):
        results = pool.starmap(procedure, itr)
        return results


def procedure(df, dataset, class_name, column, seed):
    features = list(df.columns)
    features.remove(class_name)

    # inject missing values in the df, with different percentages
    df_list_no_class = dirty_single_column(df[features], column, class_name, seed)

    results_experiment = dict()
    column_profile = ()
    for i, df_missing in enumerate(df_list_no_class):
        column_type = df[column].dtype

        imputed_datasets = []
        #print("starting imputation ", i)
        if column_type in ["int64", "float64"]:
            column_profile = get_features_num(df_missing, column)
            # impute the numerical column with all the imputation methods
            for imp_method in imp_methods_num:
                #print(imp_method)
                current_df = df_missing.copy()
                imputed_df = impute_missing_column(current_df, imp_method,
                                                   column)
                imputed_df = encoding_categorical_variables(imputed_df)
                imputed_df[class_name] = df[class_name]
                imputed_datasets.append(imputed_df)

        if column_type in ["bool", "object"]:
            column_profile = get_features_cat(df_missing, column)
            # impute the categorical column with all the imputation methods
            for imp_method in imp_methods_cat:
                #print(imp_method)
                current_df = df_missing.copy()
                imputed_df = impute_missing_column(current_df, imp_method,
                                                   column)
                imputed_df = encoding_categorical_variables(imputed_df)
                imputed_df[class_name] = df[class_name]
                imputed_datasets.append(imputed_df)

        ml_results = dict()
        # for each ml algorithm and imputed dataset, compute the corresponding score
        for ml_method in ml_methods:
            #print("starting ", ml_method)
            scores = []
            for imputed_df in imputed_datasets:
                new_features = list(imputed_df.columns)
                new_features.remove(class_name)
                param = df_hyper[
                    np.logical_and(df_hyper["ml_method"] == ml_method,
                                   df_hyper["dataset"] == dataset)][
                    "best_parameter"].values[0]
                ml_score = classification(imputed_df[new_features],
                                          imputed_df[class_name], ml_method,
                                          param)
                scores.append(ml_score)
            ml_results[ml_method] = scores

        results_experiment[i] = [column_profile, ml_results]

    return results_experiment


def write_file(dataset, column, experiment, file):
    for missing_perc in range(10): # there are ten missing percentages
        results_missing_perc = experiment[missing_perc]
        column_profile = results_missing_perc[0]
        ml_results = results_missing_perc[1]

        for ml_index, ml_method in enumerate(ml_methods):
            new_line = dataset + "," + column + ","
            for val in column_profile:
                new_line += str(val) + ","
            new_line += ml_method + ","
            for score in ml_results[ml_method]:
                new_line += str(score) + ","
            new_line = new_line[:-1]
            new_line += "\n"
            file.write(new_line)

def main(reduced_df=False):
    path_datasets = "Datasets/CSV/"
    # sempre multipli
    n_instances_tot = 8
    n_parallel_jobs = 8

    files_numerical = []
    files_categorical = []
    for i in range(n_parallel_jobs):
        file_num = open(f"experiment_{i+1}_numerical.csv","w")
        file_num.write(
            "name,column_name,n_tuples,missing_perc,uniqueness," +
            "min,max,mean,median,std,skewness,kurtosis,mad," +
            "iqr,p_min,p_max,k_min,k_max,s_min,s_max,entropy," +
            "density,ml_algorithm,impute_standard,impute_mean," +
            "impute_median,impute_random,impute_knn,impute_mice," +
            "impute_linear_regression,impute_random_forest,impute_cmeans\n")
        files_numerical.append(file_num)

        file_cat = open(f"experiment_{i+1}_categorical.csv","w")
        file_cat.write(
            "name,column_name,n_tuples,missing_perc,constancy,imbalance," +
            "uniqueness,unalikeability,entropy,density,mean_char,std_char,skewness_char," +
            "kurtosis_char,min_char,max_char,ml_method,impute_standard," +
            "impute_mode,impute_random,impute_knn,impute_mice,impute_logistic_regression," +
            "impute_random_forest,impute_kproto\n"
        )
        files_categorical.append(file_cat)

    # this files saves the seeds used for each column in the experiments, for reproducibility
    file_seeds = open("seeds.csv", "w")
    line = "name,column_name,"
    for i in range(n_parallel_jobs):
        line += f"seed_{i},"
    line = line[:-1]
    line += "\n"
    file_seeds.write(line)

    # here starts the main loop on datasets and columns
    # datasets = ["default of credit card clients", "frogs", "mushrooms", "ringnorm"]
    for dataset in datasets:
        print("------------" + dataset + "------------")
        df = get_dataset("src/" + path_datasets,dataset + ".csv")
        class_name = df.columns[-1]

        # feature selection
        # df_fs, _, _, _, _ = feature_selection_univariate(df, class_name, perc_num=50, perc_cat=60)
        df_corr_removed = remove_corr(df, class_name, threshold=0.8)
        df_fs = fixed_fs_univariate(df_corr_removed, class_name)

        columns = list(df_fs.columns)
        columns.remove(class_name)
        for column in columns:
            print("ANALYZING ", column)
            if not reduced_df:
                experiments = parallel_exec(df, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds)
            else:
                experiments = parallel_exec(df_fs, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds)

            # write the results of the different experiments in the corresponding files
            for i, experiment in enumerate(experiments):
                if df[column].dtype in ["int64","float64"]:
                    write_file(dataset, column, experiment, files_numerical[i])
                else:
                    write_file(dataset, column, experiment, files_categorical[i])

    # closing files
    for i in range(len(files_numerical)):
        files_numerical[i].close()
        files_categorical[i].close()

    file_datasets.close()
    file_imp_methods_cat.close()
    file_imp_methods_num.close()
    file_ml_methods.close()
    file_seeds.close()

if __name__ == "__main__":
    main(reduced_df=True)
