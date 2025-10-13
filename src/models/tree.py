from sklearn.tree import DecisionTreeClassifier



def tree_model(cfg, model_cfg):
    # receives: tree confg and global_cfg
    tree = DecisionTreeClassifier(
        criterion=model_cfg.criterion,
        max_depth=model_cfg.max_depth,    
        random_state=cfg.random_state
    )
    return tree