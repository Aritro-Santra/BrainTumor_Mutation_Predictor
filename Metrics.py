import torch


def emr(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    n = len(y_true)
    row_indicators = torch.all(y_true == y_pred, dim=1)  # dim = 1 will check for equality along rows
    exact_match_count = torch.sum(row_indicators)
    return exact_match_count / n


def one_zero_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    n = len(y_true)
    row_indicators = torch.logical_not(
        torch.all(y_true == y_pred, dim=1))  # dim = 1 will check for equality along rows.
    not_equal_count = torch.sum(row_indicators)
    return not_equal_count / n


def hamming_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
        XOR TT for reference -
        A  B   Output
        0  0    0
        0  1    1
        1  0    1
        1  1    0
    """
    hl_num = torch.sum(torch.logical_xor(y_true, y_pred))
    shape = y_true.shape
    hl_den = torch.prod(torch.tensor(shape))

    return hl_num / hl_den


def example_based_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # compute true positives using the logical AND operator
    numerator = torch.sum(torch.logical_and(y_true, y_pred), dim=1)

    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = torch.sum(torch.logical_or(y_true, y_pred), dim=1)
    instance_accuracy = numerator / denominator

    avg_accuracy = torch.mean(instance_accuracy)
    return avg_accuracy


def example_based_precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    precision = TP/ (TP + FP)
    """

    # Compute True Positive
    precision_num = torch.sum(torch.logical_and(y_true, y_pred), dim=1)

    # Total number of pred true labels
    precision_den = torch.sum(y_pred, dim=1)

    # precision averaged over all training examples
    avg_precision = torch.mean(precision_num / precision_den)

    return avg_precision


def label_based_macro_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # axis = 0 computes true positives along columns i.e labels
    l_acc_num = torch.sum(torch.logical_and(y_true, y_pred), dim=0)

    # axis = 0 computes true postive + false positive + false negatives along columns i.e labels
    l_acc_den = torch.sum(torch.logical_or(y_true, y_pred), dim=0)

    # compute mean accuracy across labels.
    return torch.mean(l_acc_num / l_acc_den)


def label_based_macro_precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # axis = 0 computes true positive along columns i.e labels
    l_prec_num = torch.sum(torch.logical_and(y_true, y_pred), dim=0)

    # axis = computes true_positive + false positive along columns i.e labels
    l_prec_den = torch.sum(y_pred, dim=0)

    # compute precision per class/label
    l_prec_per_class = l_prec_num / l_prec_den

    # macro precision = average of precision across labels.
    l_prec = torch.mean(l_prec_per_class)
    return l_prec


def label_based_macro_recall(y_true, y_pred):
    # compute true positive along axis = 0 i.e labels
    l_recall_num = torch.sum(torch.logical_and(y_true, y_pred), dim=0)

    # compute true positive + false negatives along axis = 0 i.e columns
    l_recall_den = torch.sum(y_true, dim=0)

    # compute recall per class/label
    l_recall_per_class = l_recall_num / l_recall_den

    # compute macro averaged recall i.e recall averaged across labels.
    l_recall = torch.mean(l_recall_per_class)
    return l_recall


def label_based_micro_accuracy(y_true, y_pred):
    # sum of all true positives across all examples and labels
    l_acc_num = torch.sum(torch.logical_and(y_true, y_pred))

    # sum of all tp+fp+fn across all examples and labels.
    l_acc_den = torch.sum(torch.logical_or(y_true, y_pred))

    # compute mirco averaged accuracy
    return l_acc_num / l_acc_den


def label_based_micro_precision(y_true, y_pred):
    # compute sum of true positives (tp) across training examples
    # and labels.
    l_prec_num = torch.sum(torch.logical_and(y_true, y_pred))

    # compute the sum of tp + fp across training examples and labels
    l_prec_den = torch.sum(y_pred)

    # compute micro-averaged precision
    return l_prec_num / l_prec_den


def label_based_micro_recall(y_true, y_pred):
    # compute sum of true positives across training examples and labels.
    l_recall_num = torch.sum(torch.logical_and(y_true, y_pred))
    # compute sum of tp + fn across training examples and labels
    l_recall_den = torch.sum(y_true)

    # compute mirco-average recall
    return l_recall_num / l_recall_den


def f1_score(y_true, y_pred):
    """
    Calculate F1 score
    y_true: true value
    y_pred: predicted value
    """
    epsilon = 1e-7

    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return (2 * precision * recall) / (precision + recall + epsilon)


if __name__ == "__main__":
    # Example
    gt = torch.tensor([[1, 0, 1, 1, 0],
                       [1, 1, 0, 1, 0],
                       [1, 0, 0, 1, 0],
                       [1, 1, 1, 0, 1],
                       [0, 1, 0, 1, 1]],
                      dtype=torch.float, requires_grad=True)
    pred = torch.tensor([[1, 0, 1, 0, 1],
                         [1, 0, 0, 0, 1],
                         [1, 1, 0, 1, 0],
                         [1, 1, 1, 0, 1],
                         [0, 1, 1, 0, 1]],
                        dtype=torch.float, requires_grad=True)
    print("Testing different metrics")
    print("Input tensor 1")
    print(gt)
    print("Input tensor 2")
    print(pred)
    print("EMR: ", emr(gt, pred).item())
    print("1/0 Loss: ", one_zero_loss(gt, pred))
    hl_value = hamming_loss(gt, pred)
    print(f"Hamming Loss: {hl_value}")
    ex_based_accuracy = example_based_accuracy(gt, pred)
    print(f"Example Based Accuracy: {ex_based_accuracy}")
    ex_based_precision = example_based_precision(gt, pred)
    print(f"Example Based Precision: {ex_based_precision}")
    lb_macro_acc_val = label_based_macro_accuracy(gt, pred)
    print(f"Label Based Macro Accuracy: {lb_macro_acc_val}")
    lb_macro_precision_val = label_based_macro_precision(gt, pred)
    print(f"Label Based Precision: {lb_macro_precision_val}")
    lb_macro_recall_val = label_based_macro_recall(gt, pred)
    print(f"Label Based Recall: {lb_macro_recall_val}")
    lb_micro_acc_val = label_based_micro_accuracy(gt, pred)
    print(f"Label Based Micro Accuracy: {lb_micro_acc_val}")
    lb_micro_prec_val = label_based_micro_precision(gt, pred)
    print(f"Label Based Micro Precision: {lb_micro_prec_val}")
    lb_micro_recall_val = label_based_micro_recall(gt, pred)
    print(f"Label Based Micro Recall: {lb_micro_recall_val}")
    f1_score = f1_score(gt, pred)
    print(f"F1 Score: {f1_score}")
