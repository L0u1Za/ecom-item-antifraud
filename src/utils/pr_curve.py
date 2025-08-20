from torchmetrics.classification import BinaryPrecisionRecallCurve


def calc_best_threshold(logits, targets):
    pr_curve = BinaryPrecisionRecallCurve(num_tasks=1)
    precision, recall, thresholds = pr_curve(logits, targets)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = f1.argmax()
    best_thr = thresholds[best_idx]
    return best_thr
