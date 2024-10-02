import pandas as pd

if __name__ == '__main__':

    """
    this script was used at the beginning for the selection of the dataset
    during the Knowledge Base enrichment process. Datasets were selected based on
    number of classes, number of rows and columns, number of categorical and numerical
    features, providing a rich set of dataset with varying characteristics.
    """
    path = "Data/"
    name = "dataset_classification.csv"
    df = pd.read_csv(path + name)

    selected_columns = ['Nome', 'Numero Classi', 'Numero Righe', 'Numero Colonne', '# Numerical', '# Categorical']
    df = df[selected_columns]

    features = ['Numero Classi', 'Numero Righe', 'Numero Colonne', '# Numerical', '# Categorical']

    dataset_set = set()
    for feature in features:

        sorted_df = df.sort_values(feature)

        selected_low = (sorted_df['Nome'][:3]).to_list()
        for dataset in selected_low:
            dataset_set.add(dataset)

        selected_high = (sorted_df['Nome'][-3:]).tolist()
        for dataset in selected_high:
            dataset_set.add(dataset)

    print(dataset_set)
    file = open(path + "selected_classification.csv", "w")
    file.write('Nome,Numero Classi,Numero Righe,Numero Colonne,# Numerical,# Categorical\n')
    for dataset in dataset_set:
        row = df.loc[df['Nome'] == dataset,:].values.flatten()
        for cell in row:
            file.write(str(cell) + ',')
        file.write('\n')

    file.close()





