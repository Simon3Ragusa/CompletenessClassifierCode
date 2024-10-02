import pandas as pd

def get_dataset(path: str, name: str):
    """
    Function that reads a csv dataset from the provided path. For some datasets, the types
    of the column must be hard-coded as pandas can assume wrong types in some cases
    :param path: path from which extract the dataset
    :param name: name of the dataset
    :return: a Pandas dataframe
    """
    if name == 'electricity-normalized.csv':
        types = {'date':float,'day':str,'period':float,'nswprice':float,'nswdemand':float,'vicprice':float,'vicdemand':float,'transfer':float,'class':str}
        df = pd.read_csv(path+name, dtype=types)
    elif name == 'visualizing_soil.csv':
        types = {'northing':float,'easting':float,'resistivity':float,'isns':str,'binaryClass':str}
        df = pd.read_csv(path+name, dtype=types)
    elif name == 'consumer.csv':
        types = {"ProductID":int,"ProductCategory":str,"ProductBrand":str,"ProductPrice":float,"CustomerAge":float,"CustomerGender":str,"PurchaseFrequency":float,"CustomerSatisfaction":float,"PurchaseIntent":str}
        df = pd.read_csv(path+name, dtype=types)
    elif name == 'student.csv':
        types = {"StudentID":int, "Age":float, "Gender":str,"Ethnicity":str,"ParentalEducation":str,"StudyTimeWeekly":float,"Absences":float,"Tutoring":str,"ParentalSupport":str,"Extracurricular":str,"Sports":str,"Music":str,"Volunteering":str,"GPA":float,"GradeClass":str}
        df = pd.read_csv(path + name, dtype=types)
    else:
        df = pd.read_csv(path + name)
    return df
