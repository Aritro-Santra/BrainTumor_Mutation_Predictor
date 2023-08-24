import torch
from torchmetrics.classification import MultilabelStatScores, MultilabelAccuracy, MultilabelPrecision, MultilabelRecall
from torchmetrics.classification import MultilabelExactMatch, MultilabelF1Score, MultilabelConfusionMatrix, MultilabelHammingDistance


def emr(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    exact_matcher = MultilabelExactMatch(num_labels=5)
    exact_match_ratio = exact_matcher(y_pred, y_true)

    # n = len(y_true)
    # row_indicators = torch.all(y_true == y_pred, dim=1)  # dim = 1 will check for equality along rows
    # exact_match_count = torch.sum(row_indicators)
    # return exact_match_count / n
    return exact_match_ratio


def one_zero_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    n = len(y_true)
    row_indicators = torch.logical_not(
        torch.all(y_true == y_pred, dim=1))  # dim = 1 will check for equality along rows.
    not_equal_count = torch.sum(row_indicators)
    return not_equal_count / n


def hamming_distance(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
        XOR TT for reference -
        A  B   Output
        0  0    0
        0  1    1
        1  0    1
        1  1    0
    """
    function = MultilabelHammingDistance(num_labels=5)
    return function(y_true, y_pred).item()


def multi_label_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    function = MultilabelAccuracy(num_labels=2)
    return function(y_pred, y_true)


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

    # axis = 0 computes true positive + false positive + false negatives along columns i.e labels
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


def label_based_macro_recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # compute true positive along axis = 0 i.e labels
    l_recall_num = torch.sum(torch.logical_and(y_true, y_pred), dim=0)

    # compute true positive + false negatives along axis = 0 i.e columns
    l_recall_den = torch.sum(y_true, dim=0)

    # compute recall per class/label
    l_recall_per_class = l_recall_num / l_recall_den

    # compute macro averaged recall i.e recall averaged across labels.
    l_recall = torch.mean(l_recall_per_class)
    return l_recall


def label_based_micro_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # sum of all true positives across all examples and labels
    l_acc_num = torch.sum(torch.logical_and(y_true, y_pred))

    # sum of all tp+fp+fn across all examples and labels.
    l_acc_den = torch.sum(torch.logical_or(y_true, y_pred))

    # compute mirco averaged accuracy
    return l_acc_num / l_acc_den


def label_based_micro_precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # compute sum of true positives (tp) across training examples
    # and labels.
    l_prec_num = torch.sum(torch.logical_and(y_true, y_pred))

    # compute the sum of tp + fp across training examples and labels
    l_prec_den = torch.sum(y_pred)

    # compute micro-averaged precision
    return l_prec_num / l_prec_den


def label_based_micro_recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # compute sum of true positives across training examples and labels.
    l_recall_num = torch.sum(torch.logical_and(y_true, y_pred))
    # compute sum of tp + fn across training examples and labels
    l_recall_den = torch.sum(y_true)

    # compute mirco-average recall
    return l_recall_num / l_recall_den


def multi_label_precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    function = MultilabelPrecision(num_labels=2)
    return function(y_pred, y_true)


def multi_label_recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    function = MultilabelRecall(num_labels=2)
    return function(y_pred, y_true)


def multi_label_stat_scores(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    function = MultilabelStatScores(num_labels=5)
    return function(y_pred, y_true)


def multi_label_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    function = MultilabelConfusionMatrix(num_labels=5)
    return function(y_pred, y_true)


def f1_score(y_true, y_pred):
    """
    Calculate F1 score
    y_true: true value
    y_pred: predicted value
    """
    # epsilon = 1e-7
    #
    # true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    # possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    # recall = true_positives / (possible_positives + epsilon)
    # predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    # precision = true_positives / (predicted_positives + epsilon)
    # return (2 * precision * recall) / (precision + recall + epsilon)
    function = MultilabelF1Score(num_labels=2)
    return function(y_pred, y_true)


if __name__ == "__main__":
    # Example with a batch of 5 binary vectors
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
    hl_value = hamming_distance(gt, pred)
    print(f"Hamming Loss: {1 - hl_value}")
    accuracy = multi_label_accuracy(gt, pred)
    print(f"Multi label Accuracy: {accuracy}")
    precision = multi_label_precision(gt, pred)
    print(f"Multi label Precision: {precision}")
    recall = multi_label_recall(gt, pred)
    print(f"Multi label Recall: {recall}")
    f1_score = f1_score(gt, pred)
    print(f"F1 Score: {f1_score}")
    stat_scores = multi_label_stat_scores(gt, pred)
    print(f"Stat Scores:", stat_scores)
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
    # confusion_matrix = multi_label_confusion_matrix(gt, pred)
    # print("Confusion Matrix")
    # print(confusion_matrix)
