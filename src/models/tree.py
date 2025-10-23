from sklearn.tree import DecisionTreeClassifier



def TreeModel(criterion, max_depth, min_samples_split, random_state):
    tree = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,    
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    return tree