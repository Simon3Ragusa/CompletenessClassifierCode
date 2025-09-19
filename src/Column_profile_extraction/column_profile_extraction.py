from Datasets.get_dataset import get_dataset
from Feature_selection.feature_selection import feature_selection_univariate
from Column_profile_extraction.numerical import get_features_num
from Column_profile_extraction.categorical import get_features_cat
from utils import dirty_single_column


# this file is only used to check everything works correctly
def main():
    path_datasets = "../Datasets/CSV/"
    file_num = open("features_numerical_columns.csv", "w")
    file_cat = open("features_categorical_columns.csv", "w")
    file_num.write("name,column_name,n_tuples,missing_perc,uniqueness," +
                   "min,max,mean,median,std,skewness,kurtosis,mad," +
                   "iqr,p_min,p_max,k_min,k_max,s_min,s_max,entropy," +
                   "density\n")
    file_cat.write("name,column_name,n_tuples,missing_perc,constancy," +
                   "imbalance,uniqueness,unalikeability,entropy,density,mean_char," +
                   "std_char,skewness_char,kurtosis_char,min_char,max_char\n")
    file_datasets = open("../Datasets/dataset_names.txt", "r")
    datasets = file_datasets.readlines()
    datasets = [line.strip('\n\r') for line in datasets]
    # datasets = ["BachChoralHarmony"] #used for testing, must be deleted
    for dataset in datasets:
        print("----------" + dataset + "----------")
        df = get_dataset(path_datasets,dataset+".csv")
        class_name = df.columns[-1]
        #df,_,_,_,_ = feature_selection_new(df,class_name, threshold_pca=0.9, perc_cat=60)
        df_fs,_,_,_,_ = feature_selection_univariate(df,class_name, perc_num=50, perc_cat=60)
        columns = list(df_fs.columns)
        columns.remove(class_name)

        for column in columns:
            column_type = df[column].dtype
            print("analyzing column: ", column)
            print("type detected: ", column_type)
            feature_cols = list(df.columns)
            feature_cols.remove(class_name)
            # now we inject missing values into the to be imputed column, ignoring the class column
            df_list_no_class = dirty_single_column(df[feature_cols], column, class_name, 0)

            for df_missing in df_list_no_class:
                if column_type in ["float64","int64"]:
                    features = get_features_num(df_missing, column)
                    new_line = str(dataset) + "," + str(column) + ","
                    for feature in features:
                        new_line += str(feature)+","
                    new_line = new_line[:-1]
                    new_line += "\n"
                    file_num.write(new_line)

                elif column_type in ["object","bool"]:
                    features = get_features_cat(df_missing, column)
                    new_line = str(dataset) + "," + str(column) + ","
                    for feature in features:
                        new_line += str(feature) + ","
                    new_line = new_line[:-1]
                    new_line += "\n"
                    file_cat.write(new_line)
    file_num.close()
    file_cat.close()
    file_datasets.close()


if __name__ == "__main__":
    main()
