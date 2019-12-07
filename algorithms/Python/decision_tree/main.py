import decision_tree as dt
from mltoolkit import string_to_float
from mltoolkit import evaluate
from csvtool import load_csv
from random import seed

if __name__ == "__main__":
    seed(1)

    file_name = 'data_banknote_authentication.csv'
    data = load_csv(file_name)

    for i in range(len(data[0])):
        string_to_float(data, i)

    num_folds = 5
    max_depth = 5
    min_size = 10

    scores = evaluate(data, dt.decision_tree, num_folds, max_depth, min_size)

    print('Scores: %s' % scores)
    print('Mean accuracy: %.3f' % (sum(scores) / float(len(scores))))
