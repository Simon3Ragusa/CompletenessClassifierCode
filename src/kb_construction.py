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
# set PyTensor flags cxx to an empty string.
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

# opening files with names of datasets, ml algorithms and imputation methods
file_datasets = open("Datasets/dataset_names.txt", "r")
file_ml_methods = open("Classification/classification_methods.txt", "r")

## =========== NEW EXPERIMENTS WITH NEW IMPUTATION METHODS ============== ##
# file_imp_methods_num = open("Imputation/methods_numerical_column.txt", "r")
# file_imp_methods_cat = open("Imputation/methods_categorical_column.txt", "r")
file_imp_methods_num = open("Imputation/new_methods_numerical_column.txt", "r")
file_imp_methods_cat = open("Imputation/new_methods_categorical_column.txt", "r")

datasets = file_datasets.readlines()
 # removing adult dataset for now
ml_methods = file_ml_methods.readlines()
imp_methods_num = file_imp_methods_num.readlines()
imp_methods_cat = file_imp_methods_cat.readlines()

datasets = [line.strip('\n\r') for line in datasets]
ml_methods = [line.strip('\n\r') for line in ml_methods]
imp_methods_num = [line.strip('\n\r') for line in imp_methods_num]
imp_methods_cat = [line.strip('\n\r') for line in imp_methods_cat]

# this dataframe contains the value of the parameters to train the ml algorithms
df_hyper = pd.read_csv("Hyperparameter_tuning/hyperparameters.csv")

done_ds = ['abalone', 'BachChoralHarmony', 'bank', 'cancer', 'car', 'consumer', 'dataset_188_kropt', 'default of credit card clients', 'diabetic', 'drug', 'electricity-normalized', 'fried', 'frogs', 'german']
datasets = [ds for ds in datasets if ds not in done_ds] 
# generate seeds for the different parallel jobs
def generate_seed(n_seed, n_elements):
    seed = []
    seeds = []
    for r in range(0, n_seed):
        for i in range(0, n_elements):
            seed.append(int(np.random.randint(0, 100)))
        seeds.append(seed)
        seed = []
    return seeds

# execute the experiments in parallel
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

    # create the iterator for the parallel execution
    itr = zip(repeat(df), repeat(dataset), repeat(class_name), repeat(column), seed)

    # starts the parallel experiments on the column
    with (Pool(processes=n_parallel_jobs) as pool):
        results = pool.starmap(procedure, itr)
        return results


# procedure for the experiments on a specific column
def procedure(df, dataset, class_name, column, seed):
    features = list(df.columns)
    features.remove(class_name)

    # inject missing values in the df, with different percentages. This data frame contains different versions of the column with missing values (different percentages)
    df_list_no_class = dirty_single_column(df[features], column, class_name, seed)

    # Initialize the results dictionary
    results_experiment = dict()

    column_profile = ()
    
    for i, df_missing in enumerate(df_list_no_class):
        column_type = df[column].dtype

        imputed_datasets = []
        # print("Starting imputation on first dirty dataset ", i)
        if column_type in ["int64", "float64"]:

            # Profile extraction for numerical column with missing values
            column_profile = get_features_num(df_missing, column)

            # impute the numerical column with all the imputation methods
            for imp_method in imp_methods_num:
                print("[", imp_method, "]")
                current_df = df_missing.copy()
                imputed_df = impute_missing_column(current_df, imp_method,
                                                column)
                imputed_df = encoding_categorical_variables(imputed_df)
                # add the class column back to the imputed dataframe
                imputed_df[class_name] = df[class_name]
                imputed_datasets.append(imputed_df)
                print("Imputation with method ", imp_method, " completed.")

        if column_type in ["bool", "object"]:
            column_profile = get_features_cat(df_missing, column)
            # impute the categorical column with all the imputation methods
            for imp_method in imp_methods_cat:
                print("[", imp_method, "]")   
                current_df = df_missing.copy()
                
                imputed_df = impute_missing_column(current_df, imp_method,
                                                column)
                # print("Imputed dataset shape: ", imputed_df.shape)
                # print("Inputed column unique values: ", imputed_df[column].unique())
                imputed_df = encoding_categorical_variables(imputed_df)

                # add the class column back to the imputed dataframe
                imputed_df[class_name] = df[class_name]
                imputed_datasets.append(imputed_df)
                print("Imputation with method ", imp_method, " completed.")
                # print("Imputed dataset shape: ", imputed_df.shape)
                # print("Inputed dataset columns: ", imputed_df.columns)
                # print("")
        # for imputed in imputed_datasets:
        #     print("Type of imputed dataset: ", type(imputed))
        
        ml_results = dict()
        
        print("Starting ML evaluation...")
        for ml_method in ml_methods:
            print("starting ", ml_method)
            scores = []
            for imputed_df in imputed_datasets:
                # print("Imputed dataset shape: ", imputed_df.shape)
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

        print("/=======================================================/")
        print("Experiment for iteration ", i, " completed.")
        print("/=======================================================/")
    return results_experiment


def write_file(dataset, column, experiment, file):
    print("Writing results for dataset ", dataset, " column ", column)
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
    print("Starting knowledge base construction...")
    # print imputation and ml methods used
    print("Imputation methods for numerical columns: ", imp_methods_num)
    print("Imputation methods for categorical columns: ", imp_methods_cat)
    print("ML methods: ", ml_methods)

    path_datasets = "Datasets/CSV/"
    new_exp_path = "NewExp/"
    # sempre multipli
    n_instances_tot = 8
    n_parallel_jobs = 8

    # Opening file to save the results (in the new experiments folder)
    files_numerical = []
    files_categorical = []
    for i in range(n_parallel_jobs):
        file_num = open(f"{new_exp_path}experiment_{i+1}_numerical.csv","w")
        file_num.write(
            "name,column_name,n_tuples,missing_perc,uniqueness," +
            "min,max,mean,median,std,skewness,kurtosis,mad," +
            "iqr,p_min,p_max,k_min,k_max,s_min,s_max,entropy," +
            "density,ml_algorithm,impute_standard,impute_mean," +
            "impute_median,impute_random,impute_knn,impute_mice," +
            "impute_linear_regression,impute_random_forest,impute_cmeans\n"
        )
        print("Numerical file header written.")
        files_numerical.append(file_num)

        file_cat = open(f"{new_exp_path}experiment_{i+1}_categorical.csv","w")
        file_cat.write(
            "name,column_name,n_tuples,missing_perc,constancy,imbalance," +
            "uniqueness,unalikeability,entropy,density,mean_char,std_char,skewness_char," +
            "kurtosis_char,min_char,max_char,ml_method,impute_standard," +
            "impute_mode,impute_random,impute_knn,impute_mice,impute_logistic_regression," +
            "impute_random_forest,impute_kproto\n"
        )
        print("Categorical file header written.")
        files_categorical.append(file_cat)

    # # Test write on categorial file
    # print("Test write on categorical file.")
    # print("File name: ", files_categorical[0].name)
    # test_line = "test_dataset,test_column,1000,0.1,0.5,0.3,0.2,0.4,1.5,0.6,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,ml_test,impute_standard_test,impute_mode_test,impute_random_test,impute_knn_test,impute_mice_test,impute_logistic_regression_test,impute_random_forest_test,impute_kproto_test\n"
    # files_categorical[0].write(test_line)
    # print("Test write completed.")

    # this files saves the seeds used for each column in the experiments, for reproducibility
    file_seeds = open(f"{new_exp_path}seeds.csv", "w")
    line = "name,column_name,"
    for i in range(n_parallel_jobs):
        line += f"seed_{i},"
    line = line[:-1]
    line += "\n"
    file_seeds.write(line)

    # here starts the main loop on datasets and columns
    print("Datasets to analyze: ", datasets)
    errors = []
    for dataset in datasets:  # removing adult dataset for now
        print("------------" + dataset + "------------")
        df = get_dataset(path_datasets,dataset + ".csv")
        class_name = df.columns[-1]

        # feature selection
        # df_fs, _, _, _, _ = feature_selection_univariate(df, class_name, perc_num=50, perc_cat=60)
        try:
            df_corr_removed = remove_corr(df, class_name, threshold=0.8)
            df_fs = fixed_fs_univariate(df_corr_removed, class_name)
        except Exception as e:
            print(f"Error during feature selection for dataset {dataset}: {e}")
            errors.append((dataset, "feature_selection", str(e)))
            continue
        columns = list(df_fs.columns)
        columns.remove(class_name)
        print("Columns selected after removing correlated features: ", columns)
        for column in columns:
            try:
                print("ANALYZING ", column)
                if not reduced_df:
                    # print("Using full dataset for experiments.")
                    experiments = parallel_exec(df, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds)
                else:
                    # print("Using reduced dataset for experiments.")
                    experiments = parallel_exec(df_fs, dataset, class_name, column, n_parallel_jobs, n_instances_tot, file_seeds)
                    # print("Experiments on column ", column, " completed.")
                    # print("Experiments results: ", experiments)

                # write the results of the different experiments in the corresponding files
                for i, experiment in enumerate(experiments):
                    if df[column].dtype in ["int64","float64"]:
                        print("Writing results on numerical file: ", files_numerical[i].name)
                        try:
                            write_file(dataset, column, experiment, files_numerical[i])
                            print("Write completed.")
                        except Exception as e:
                            print(f"Error writing to numerical file {files_numerical[i].name}: {e}")
                    else:
                        print("Writing results on categorical file: ", files_categorical[i].name)
                        try:
                            write_file(dataset, column, experiment, files_categorical[i])
                            print("Write completed.")
                        except Exception as e:
                            print(f"Error writing to categorical file {files_categorical[i].name}: {e}")
            except Exception as e:
                print(f"Error in main loop for dataset {dataset}, column {column}: {e}")
                errors.append((dataset, column, str(e)))
                continue
        
    # print errors if any
    if errors:
        with open(f"{new_exp_path}errors_log.txt", "w") as error_file:
            for err in errors:
                error_file.write(f"Dataset: {err[0]}, Column: {err[1]}, Error: {err[2]}\n")
        print(f"Errors logged in {new_exp_path}errors_log.txt")

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
