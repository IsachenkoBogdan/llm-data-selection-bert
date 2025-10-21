from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }
