
# Importing library
import os

# Getting all the arff files from the current directory
arff_path = 'Arff'
files = [arff for arff in os.listdir(arff_path) if arff.endswith(".arff")]

# Function for converting arff list to csv list
def toCsv(text):
    data = False
    header = ""
    new_content = []
    for line in text:
        if not data:
            if "@ATTRIBUTE" in line or "@attribute" in line:
                attributes = line.split()
                if("@attribute" in line):
                    attri_case = "@attribute"
                else:
                    attri_case = "@ATTRIBUTE"
                column_name = attributes[attributes.index(attri_case) + 1]
                header = header + column_name + ","
            elif "@DATA" in line or "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                new_content.append(header)
        else:
            new_content.append(line)
    return new_content

if __name__ == '__main__':
    # Main loop for reading and writing files
    csv_path = 'CSV'
    for file in files:
        with open(arff_path + '/' + file, "r") as inFile:
            content = inFile.readlines()
            name, ext = os.path.splitext(inFile.name)
            new = toCsv(content)
            new_path = os.path.join(csv_path, name[5:])
            with open(new_path + ".csv", "w") as outFile:
                outFile.writelines(new)