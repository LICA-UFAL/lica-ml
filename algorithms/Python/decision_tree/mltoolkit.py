import random

def string_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def cross_validation_split(dataset, num_folds):
    """
    Split a data into k folds
    """
    dataset_split = list()
    dataset_ = list(dataset)
    fold_size = int(len(dataset) / num_folds)
    
    for i in range(num_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_))
            fold.append(dataset_.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if (actual[i] == predicted[i]):
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate(dataset, algorithm, num_folds, *args):
    folds = cross_validation_split(dataset=dataset, num_folds=num_folds)
    scores = list()

    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        
        for row in fold:
            row_ = list(row)
            test_set.append(row_)
            row_[-1] = None
        
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        acc = accuracy(actual, predicted)
        scores.append(acc)
    return scores
