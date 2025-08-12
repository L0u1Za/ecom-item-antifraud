def k_fold_cross_validation(data, k, stratified=False, groups=None):
    from sklearn.model_selection import KFold, StratifiedKFold

    if stratified and groups is not None:
        raise ValueError("Cannot use both stratified and groups at the same time.")

    if stratified:
        skf = StratifiedKFold(n_splits=k)
        for train_index, test_index in skf.split(data, data['target']):
            yield train_index, test_index
    elif groups is not None:
        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(data, groups):
            yield train_index, test_index
    else:
        kf = KFold(n_splits=k)
        for train_index, test_index in kf.split(data):
            yield train_index, test_index