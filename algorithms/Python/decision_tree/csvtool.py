from csv import reader

def load_csv(file_name):
    file = open(file_name, "r")
    lines = reader(file)
    dataset = list(lines)

    return dataset
