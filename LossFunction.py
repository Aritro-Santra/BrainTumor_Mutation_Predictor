import torch
import torch.nn as nn
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torchmetrics.classification import MultilabelHammingDistance, MultilabelRankingLoss


class MultiLabelWCELoss(nn.Module):
    def __init__(self, dataframe, device, epsilon=1e-7, test=False):
        super(MultiLabelWCELoss, self).__init__()
        self.epsilon = epsilon
        self.test = test
        if not self.test:
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
            self.positive_weights = torch.zeros((1, 5)).to(device)
            self.negative_weights = torch.zeros((1, 5)).to(device)

            for idx, label in enumerate(self.labels):
                self.positive_weights[0, idx] = num / (2 * sum(dataframe[label] == 1))
                self.negative_weights[0, idx] = num / (2 * sum(dataframe[label] == 0))
        else:
            # Calculating class weights for each label
            self.positive_weights = self.epsilon + torch.rand((1, 5)).to(device)
            self.negative_weights = self.epsilon + torch.rand((1, 5)).to(device)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
            Multi-label cross-entropy
            * Required "Wp", "Wn" as positive & negative class-weights
            y_true: true value as binary vector
            y_pred: predicted value as binary vector
        """
        y_true = y_true + self.epsilon
        y_pred = y_pred + self.epsilon
        if self.test:
            print("Positive weights shape: ", self.positive_weights.shape)
            print("Y True shape: ", y_true.shape)
            print("Negative weights shape: ", self.negative_weights.shape)
            print("Y Pred shape: ", y_pred.shape)
            a1 = self.positive_weights * y_true
            b1 = torch.log(y_pred)
            print("Positive_Weights * Y_True = ", a1)
            print("Log(Y_Pred) = ", b1)
            print("(Positive_Weights*Y_True)*Log(Y_Pred)+epsilon = ", a1 * b1)

            a2 = self.negative_weights * (1 + self.epsilon - y_true)
            b2 = torch.log(1 + self.epsilon - torch.log(y_pred))
            print("1 - y_pred:\n", 1 + self.epsilon - torch.log(y_pred))
            print("Negative_Weights * (1 - Y_True) = ", a2)
            print("Log(1 - Y_Pred) = ", b2)
            print("(Negative_Weights*(1 - Y_True))*Log(1 - Y_Pred)+epsilon = ", a2 * b2)

            loss = -torch.sum(torch.add((a1 * b1), (a2 * b2)), dim=1)
            print("Loss Tensor: ", loss)
            return torch.mean(loss)

        else:
            a1 = self.positive_weights * y_true
            b1 = torch.log(y_pred)

            a2 = self.negative_weights * (1 + self.epsilon - y_true)
            b2 = torch.log(1 + self.epsilon - torch.log(y_pred))

            loss = -torch.sum(torch.add((a1 * b1), (a2 * b2)), dim=1)

            return torch.mean(loss)

            # for i, key in enumerate(self.labels):
            #     first_term = self.class_weights['positive_weights'][key] * torch.select(y_true, 1, i)\
            #                  * torch.log(torch.select(y_pred, 1, i) + self.epsilon)
            #     second_term = self.class_weights['negative_weights'][key] * (1 - torch.select(y_true, 1, i)
            #                                                                  * torch.log(1 - torch.select(y_pred, 1, i))
            #                                                                  + self.epsilon)
            #     loss -= torch.sum(first_term + second_term)


class EigenLoss(nn.Module):
    def __init__(self, threshold: float = 0.85, test: bool = False):
        super(EigenLoss, self).__init__()
        self.threshold = threshold
        self.test = test

    def build_graph(self, x: torch.Tensor) -> torch.Tensor:
        y = 1 - x
        batch_size = x.shape[0]
        x_list = list(x.detach().cpu().numpy())
        super_mat = None
        for idx, mutation_vector in enumerate(x_list):
            if mutation_vector[0] == 0:
                mat = torch.select(y, 0, idx)
            else:
                mat = torch.select(x, 0, idx)
            for mutation_value in mutation_vector[1:]:
                if mutation_value == 0:
                    mat = torch.row_stack((mat, torch.select(y, 0, idx)))
                else:
                    mat = torch.row_stack((mat, torch.select(x, 0, idx)))
            if idx == 0:
                super_mat = mat
            else:
                super_mat = torch.row_stack((super_mat, mat))
        super_mat = torch.reshape(super_mat, (batch_size, 5, 5,))
        super_mat = super_mat * 2   # All 1's becomes 2 while 0 remains 0
        super_mat = super_mat - 1   # All 0's become -1 while 2's become 1
        diag = torch.diagonal(super_mat, dim1=1, dim2=2)
        super_mat_diagonal = torch.diag_embed(diag, dim1=1, dim2=2)  # Will get a 5 x 5 diagonal matrix with elements
        # same as the diagonals of super_mat
        super_mat = super_mat - super_mat_diagonal
        return super_mat

    def calculate_laplacian(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Calculate the degree matrix
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1) - 1)

        # Calculate the Laplacian matrix
        laplacian_matrix = degree_matrix - adjacency_matrix

        return laplacian_matrix

    def select_best_eigen_values(self, eigen_values: torch.Tensor, threshold: float) -> int:
        total_eig_sum = torch.sum(eigen_values, dim=1)  # Sum the eigen values of each data
        num_values = eigen_values.shape[1]
        for k in range(1, num_values + 1):  # looping in range 1 to number of eigen values for each data
            column_eig_vals = eigen_values[:, 0:k]  # selecting the first k values from each eigen value
            temp_eig_sum = torch.sum(column_eig_vals, dim=1)
            energy = torch.abs(torch.mean(temp_eig_sum / total_eig_sum)).item()
            if energy > threshold:
                return k
        return k

    def eigen_value_similarity(self, l_pred: torch.Tensor, l_true: torch.Tensor) -> torch.Tensor:
        pred_eigvals = torch.linalg.eigvals(l_pred)
        true_eigvals = torch.linalg.eigvals(l_true)
        # print("Predicted Eigen values:", pred_eigvals)
        # print("Actual Eigen values:", true_eigvals)
        k_pred = self.select_best_eigen_values(pred_eigvals, self.threshold)
        k_true = self.select_best_eigen_values(true_eigvals, self.threshold)
        min_k = min(k_true, k_pred)
        if self.test:
            print("Minimum k selected as: ", min_k)
        distance = torch.norm(pred_eigvals[:, 0:min_k] - true_eigvals[:, 0:min_k])
        return distance

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        graph_pred = self.build_graph(y_pred)
        graph_true = self.build_graph(y_true)
        if self.test:
            print("Eigen Loss ------- Constructed Graphs:")
            print("Graph from predictions:")
            print(graph_pred)
            print("Graph from ground truth:")
            print(graph_true)

        lap_pred = self.calculate_laplacian(graph_pred)
        lap_true = self.calculate_laplacian(graph_true)

        similarity = self.eigen_value_similarity(lap_pred, lap_true)
        return torch.mean(similarity)


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        similarity = pairwise_cosine_similarity(y_true, y_pred, reduction='mean')
        return torch.mean(1 - similarity)


class HammingLoss(nn.Module):
    def __int__(self, device: torch.device):
        super(HammingLoss, self).__int__()
        self.device = device

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        distance_func = MultilabelHammingDistance(num_labels=5)
        # distance_func = distance_func.to(device=self.device)
        distance = distance_func(y_pred, y_true)
        return torch.mean(1 - distance)


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example with a batch of 5 binary vectors
    gt = torch.tensor([[1, 0, 1, 1, 0],
                       [1, 1, 0, 1, 0],
                       [1, 0, 0, 1, 0],
                       [1, 1, 1, 0, 1],
                       [0, 1, 0, 1, 1],
                       [1, 1, 0, 1, 0],
                       [1, 1, 0, 0, 1],
                       [1, 0, 0, 1, 0]],
                      dtype=torch.float, requires_grad=True).to(DEVICE)
    pred = torch.tensor([[1, 0, 1, 0, 1],
                         [1, 0, 0, 0, 1],
                         [1, 1, 0, 1, 0],
                         [1, 1, 1, 0, 1],
                         [0, 1, 1, 0, 1],
                         [1, 1, 0, 1, 0],
                         [1, 0, 0, 1, 1],
                         [1, 0, 1, 1, 0]],
                        dtype=torch.float, requires_grad=True).to(DEVICE)
    wce_loss_function = MultiLabelWCELoss(dataframe=None, device=DEVICE, test=True)
    eig_loss_function = EigenLoss(test=True)
    mul_margin_loss_function = nn.MultiLabelMarginLoss()
    cosine_loss_function = CosineLoss()
    hamming_loss_function = MultilabelHammingDistance(num_labels=5).to(device=DEVICE)
    ranking_loss_function = MultilabelRankingLoss(num_labels=5).to(device=DEVICE)

    wce_loss = wce_loss_function(gt, pred)
    eigen_loss = eig_loss_function(gt, pred)
    mul_margin_loss = mul_margin_loss_function(gt, pred.long())
    cosine_loss = cosine_loss_function(gt, pred)
    hamming_loss = 1 - hamming_loss_function(pred, gt)
    ranking_loss = ranking_loss_function(pred, gt)

    print("Multi label weighted cross entropy loss: ", wce_loss)
    print("Eigen Loss: ", eigen_loss)
    print("Multi label margin Loss: ", mul_margin_loss)
    print("Cosine Loss (1 - cosine similarity): ", cosine_loss)
    print("Hamming Loss (1 - Hamming Distance):", hamming_loss)
    print("Ranking Loss: ", ranking_loss)
    print("Final Loss:", wce_loss + eigen_loss)
