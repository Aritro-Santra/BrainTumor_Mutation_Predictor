import torch
import torch.nn as nn


class MultiLabelWCELoss(nn.Module):
    def __init__(self, dataframe, epsilon=1e-7):
        super(MultiLabelWCELoss, self).__init__()
        self.epsilon = epsilon
        # Finding number of positive and negative samples in each label
        num = len(dataframe)

        # Extracting the mutation column names which is located from index 2 to 2nd last
        self.labels = dataframe.keys()[2:-1]

        # Counting positive and negative samples (presence or absence of mutation)
        for k, label in enumerate(self.labels):
            positives = sum(dataframe[label] == 1)
            negatives = num - positives
            print('{}:\tPositive Samples: {}\t\tNegative Samples: {}'.format(label, positives, negatives))

        # Calculating class weights for each label
        self.positive_weights = torch.zeros((5, 1))
        self.negative_weights = torch.zeros((5, 1))

        for idx, label in enumerate(self.labels):
            self.positive_weights[idx, 0] = num / (2 * sum(dataframe[label] == 1))
            self.negative_weights[idx, 0] = num / (2 * sum(dataframe[label] == 0))

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
            Multi-label cross-entropy
            * Required "Wp", "Wn" as positive & negative class-weights
            y_true: true value
            y_pred: predicted value
        """

        first_term = torch.dot(self.positive_weights, y_true) * torch.log(y_pred) + self.epsilon
        second_term = torch.dot(self.negative_weights, (1 - y_true)) * torch.log(1 - y_pred) + self.epsilon
        loss = torch.sum(first_term + second_term)

        # for i, key in enumerate(self.labels):
        #     first_term = self.class_weights['positive_weights'][key] * torch.select(y_true, 1, i)\
        #                  * torch.log(torch.select(y_pred, 1, i) + self.epsilon)
        #     second_term = self.class_weights['negative_weights'][key] * (1 - torch.select(y_true, 1, i)
        #                                                                  * torch.log(1 - torch.select(y_pred, 1, i))
        #                                                                  + self.epsilon)
        #     loss -= torch.sum(first_term + second_term)

        return torch.mean(loss)
