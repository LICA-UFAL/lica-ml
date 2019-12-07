def test_split(index, value, dataset):
    """
    Split a dataset based on an attribute
    and an attribute value
    """
    left, right, = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def gini_index(groups, classes):
    """
    Calculate a gini index
    for a split dataset
    """

    # Count all samples at split point
    num_instances = float(sum([len(group) for group in groups]))

    # sum weighted Gini index for each group
    gini = 0.0

    for group in groups:
        size = float(len(group))

        # avoid divide by zero
        if (size == 0):
            continue
        score = 0.0

        # score the group based on the score for each class
        for class_value in classes:
            p = [row[-1] for row in group].count(class_value) / size
            score += p * p
        
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / num_instances)
    return gini


def get_split(dataset):
    """
    Select the best split point
    for a dataset
    """
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index=index, value=row[index], dataset=dataset)
            gini = gini_index(groups=groups, classes=class_values)
            if (gini < b_score):
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value': b_value, 'groups':b_groups}


def to_terminal(group):
    """
    Create a terinal node value
    """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    """
    Create child splits
    for a node or make terminal
    """
    left, right = node['groups']
    del(node['groups'])

    # check for a node split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left+right)
        return
    
    # check for max depth
    if (depth >= max_depth):
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    
    # process left child
    if (len(left) <= min_size):
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node=node['left'], max_depth=max_depth, min_size=min_size, depth=depth+1)
    
    # process right child
    if (len(right) <= min_size):
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node=node['right'], max_depth=max_depth, min_size=min_size, depth=depth+1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(node=root, max_depth=max_depth, min_size=min_size, depth=1)
    return root


def predict(node, row):
    """
    Make a prediction
    with a decision tree
    """
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node=node['left'], row=row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node=node['right'], row=row)
        else:
            return node['right']


def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train=train, max_depth=max_depth, min_size=min_size)
    predictions = list()
    for row in test:
        prediction = predict(node=tree, row=row)
        predictions.append(prediction)
    return(predictions)
