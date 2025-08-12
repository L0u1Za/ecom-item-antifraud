def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def precision(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum()
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

def recall(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum()
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

def log_metrics(metrics_dict):
    for key, value in metrics_dict.items():
        print(f"{key}: {value:.4f}")