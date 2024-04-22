from typing import List, Tuple


def get_confusion_matrix(
        y_true: List[int], y_pred: List[int], num_classes: int,
) -> List[List[int]]:
    """
    Generate a confusion matrix in a form of a list of lists.

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values
    :param num_classes: number of supported classes

    :return: confusion matrix
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Invalid input shapes!")

    matrix = [[0] * num_classes for _ in range(num_classes)]

    for actual, predicted in zip(y_true, y_pred):
        if actual >= num_classes or predicted >= num_classes:
            raise ValueError("Invalid prediction classes!")
        matrix[actual][predicted] += 1

    return matrix


def get_quality_factors(
        y_true: List[int],
        y_pred: List[int],
) -> Tuple[int, int, int, int]:
    """
    Calculate True Negative, False Positive, False Negative and True Positive 
    metrics basing on the ground truth and predicted lists.

    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: a tuple of TN, FP, FN, TP
    """
    matrix = get_confusion_matrix(y_true, y_pred, 2)

    return matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]


def accuracy_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the accuracy for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: accuracy score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)

    return (TP + TN) / (TP + FN + FP + TN)


def precision_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the precision for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: precision score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)

    return TP / (TP + FP)


def recall_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the recall for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: recall score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)

    return TP / (TP + FN)


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """
    Calculate the F1-score for given lists.
    :param y_true: a list of ground truth values
    :param y_pred: a list of prediction values

    :return: F1-score
    """
    TN, FP, FN, TP = get_quality_factors(y_true, y_pred)

    return 2 * TP / (2 * TP + FP + FN)
