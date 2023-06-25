import numpy as np
import pandas as pd
import os
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import warnings
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from WSIDataset import WSIDataset
from LossFunction import MultiLabelWCELoss, EigenLoss, CosineLoss
from torchmetrics.classification import MultilabelRankingLoss, MultilabelHammingDistance
from Metrics import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

warnings.filterwarnings("ignore")

# initialize the computation device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# learning parameters
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 32


def build_model(pretrained=False, fine_tune=True, num_classes=5):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features=1280, out_features=640),
        nn.Linear(in_features=640, out_features=256),
        nn.Linear(in_features=256, out_features=64),
        nn.Linear(in_features=64, out_features=num_classes)
    )

    return model


# training function
def train(model, dataloader, optimizer, criterions, metrics_dict, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    targets = []
    predictions = []
    emr_list = []
    one_zero_loss_list = []
    hamming_loss_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    # example_based_accuracy_list = []
    # example_based_precision_list = []
    # label_based_macro_accuracy_list = []
    # label_based_macro_precision_list = []
    # label_based_macro_recall_list = []
    # label_based_micro_accuracy_list = []
    # label_based_micro_precision_list = []
    # label_based_micro_recall_list = []
    f1_score_list = []
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        binarized_outputs = torch.round(outputs)
        # binarized_outputs.requires_grad = True
        # binarized_outputs.retain_grad()
        # print("GT: ", target)
        # print("Prediction: ", binarized_outputs)
        # print(binarized_outputs.grad_fn)
        # print(binarize_outputs.requires_grad)
        wce_loss = criterions[0](target, binarized_outputs)
        eigen_loss = criterions[1](target, binarized_outputs)
        ce_loss = criterions[2](target, binarized_outputs)
        mul_margin_loss = criterions[3](target, binarized_outputs.long())
        # total_loss = torch.add(wce_loss, eigen_loss)
        # total_loss = torch.add(wce_loss, ce_loss)
        # train_running_loss += total_loss.item()
        # train_running_loss += wce_loss.item()
        train_running_loss += ce_loss.item()
        # train_running_loss += mul_margin_loss.item()
        # backpropagation

        # total_loss.backward()
        # wce_loss.backward()
        ce_loss.backward()
        # mul_margin_loss.backward()
        # update optimizer parameters
        optimizer.step()
        y_true = target.detach().cpu()
        y_pred = binarized_outputs.detach().cpu()

        emr_score = emr(y_true, y_pred)
        one_zero_loss_score = one_zero_loss(y_true, y_pred)
        hamming_loss_score = hamming_distance(y_true, y_pred)
        accuracy_score = multi_label_accuracy(y_true, y_pred)
        precision_score = multi_label_precision(y_true, y_pred)
        recall_score = multi_label_recall(y_true, y_pred)
        # example_based_accuracy_score = example_based_accuracy(y_true, y_pred)
        # example_based_precision_score = example_based_precision(y_true, y_pred)
        # label_based_macro_accuracy_score = label_based_macro_accuracy(y_true, y_pred)
        # label_based_macro_precision_score = label_based_macro_precision(y_true, y_pred)
        # label_based_macro_recall_score = label_based_macro_recall(y_true, y_pred)
        # label_based_micro_accuracy_score = label_based_micro_accuracy(y_true, y_pred)
        # label_based_micro_precision_score = label_based_micro_precision(y_true, y_pred)
        # label_based_micro_recall_score = label_based_micro_recall(y_true, y_pred)
        f1_score_val = f1_score(y_true, y_pred)
        emr_list.append(emr_score)
        one_zero_loss_list.append(one_zero_loss_score)
        hamming_loss_list.append(hamming_loss_score)
        accuracy_list.append(accuracy_score)
        precision_list.append(precision_score)
        recall_list.append(recall_score)
        # example_based_accuracy_list.append(example_based_accuracy_score)
        # example_based_precision_list.append(example_based_precision_score)
        # label_based_macro_accuracy_list.append(label_based_macro_accuracy_score)
        # label_based_macro_precision_list.append(label_based_macro_precision_score)
        # label_based_macro_recall_list.append(label_based_macro_recall_score)
        # label_based_micro_accuracy_list.append(label_based_micro_accuracy_score)
        # label_based_micro_precision_list.append(label_based_micro_precision_score)
        # label_based_micro_recall_list.append(label_based_micro_recall_score)
        f1_score_list.append(f1_score_val)

        # confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
        # print(confusion_matrix)
        # print(classification_report(y_true, y_pred))
        targets.append(y_true.numpy())
        predictions.append(y_pred.numpy())
    metrics_dict['emr'] = np.asarray(emr_list).mean()
    metrics_dict['one_zero'] = np.asarray(one_zero_loss_list).mean()
    metrics_dict['hamming'] = np.asarray(hamming_loss_list).mean()
    metrics_dict['accuracy'] = np.asarray(accuracy_list).mean()
    metrics_dict['precision'] = np.asarray(precision_list).mean()
    metrics_dict['recall'] = np.asarray(recall_list).mean()
    metrics_dict['f1_score'] = np.asarray(f1_score_list).mean()

    # metrics_dict['ex_accuracy'] = np.asarray(example_based_accuracy_list).mean()
    # metrics_dict['ex_precision'] = np.asarray(example_based_precision_list).mean()
    # metrics_dict['lbl_macro_accuracy'] = np.asarray(label_based_macro_accuracy_list).mean()
    # metrics_dict['lbl_macro_precision'] = np.asarray(label_based_macro_precision_list).mean()
    # metrics_dict['lbl_macro_recall'] = np.asarray(label_based_macro_recall_list).mean()
    # metrics_dict['lbl_micro_accuracy'] = np.asarray(label_based_micro_accuracy_list).mean()
    # metrics_dict['lbl_micro_precision'] = np.asarray(label_based_micro_precision_list).mean()
    # metrics_dict['lbl_micro_recall'] = np.asarray(label_based_micro_recall_list).mean()
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    train_loss = train_running_loss / counter
    return train_loss, targets, predictions


# validation function
def validate(model, dataloader, criterions, metrics_dict, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    targets = []
    predictions = []
    emr_list = []
    one_zero_loss_list = []
    hamming_loss_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    # example_based_accuracy_list = []
    # example_based_precision_list = []
    # label_based_macro_accuracy_list = []
    # label_based_macro_precision_list = []
    # label_based_macro_recall_list = []
    # label_based_micro_accuracy_list = []
    # label_based_micro_precision_list = []
    # label_based_micro_recall_list = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            binarized_outputs = torch.round(outputs)
            wce_loss = criterions[0](target, binarized_outputs)
            eigen_loss = criterions[1](target, binarized_outputs)
            total_loss = torch.add(wce_loss, eigen_loss)
            val_running_loss += total_loss.item()
            y_true = target.detach().cpu()
            y_pred = binarized_outputs.detach().cpu()
            emr_score = emr(y_true, y_pred)
            one_zero_loss_score = one_zero_loss(y_true, y_pred)
            hamming_loss_score = hamming_distance(y_true, y_pred)
            accuracy_score = multi_label_accuracy(y_true, y_pred)
            precision_score = multi_label_precision(y_true, y_pred)
            recall_score = multi_label_recall(y_true, y_pred)

            # example_based_accuracy_score = example_based_accuracy(y_true, y_pred)
            # example_based_precision_score = example_based_precision(y_true, y_pred)
            # label_based_macro_accuracy_score = label_based_macro_accuracy(y_true, y_pred)
            # label_based_macro_precision_score = label_based_macro_precision(y_true, y_pred)
            # label_based_macro_recall_score = label_based_macro_recall(y_true, y_pred)
            # label_based_micro_accuracy_score = label_based_micro_accuracy(y_true, y_pred)
            # label_based_micro_precision_score = label_based_micro_precision(y_true, y_pred)
            # label_based_micro_recall_score = label_based_micro_recall(y_true, y_pred)
            f1_score_val = f1_score(y_true, y_pred)
            emr_list.append(emr_score)
            one_zero_loss_list.append(one_zero_loss_score)
            hamming_loss_list.append(hamming_loss_score)
            accuracy_list.append(accuracy_score)
            precision_list.append(precision_score)
            recall_list.append(recall_score)
            # example_based_accuracy_list.append(example_based_accuracy_score)
            # example_based_precision_list.append(example_based_precision_score)
            # label_based_macro_accuracy_list.append(label_based_macro_accuracy_score)
            # label_based_macro_precision_list.append(label_based_macro_precision_score)
            # label_based_macro_recall_list.append(label_based_macro_recall_score)
            # label_based_micro_accuracy_list.append(label_based_micro_accuracy_score)
            # label_based_micro_precision_list.append(label_based_micro_precision_score)
            # label_based_micro_recall_list.append(label_based_micro_recall_score)
            f1_score_list.append(f1_score_val)
            targets.append(y_true.numpy())
            predictions.append(y_pred.numpy())
        val_loss = val_running_loss / counter
        metrics_dict['emr'] = np.asarray(emr_list).mean()
        metrics_dict['one_zero'] = np.asarray(one_zero_loss_list).mean()
        metrics_dict['hamming'] = np.asarray(hamming_loss_list).mean()
        metrics_dict['accuracy'] = np.asarray(accuracy_list).mean()
        metrics_dict['precision'] = np.asarray(precision_list).mean()
        metrics_dict['recall'] = np.asarray(recall_list).mean()
        # metrics_dict['ex_accuracy'] = np.asarray(example_based_accuracy_list).mean()
        # metrics_dict['ex_precision'] = np.asarray(example_based_precision_list).mean()
        # metrics_dict['lbl_macro_accuracy'] = np.asarray(label_based_macro_accuracy_list).mean()
        # metrics_dict['lbl_macro_precision'] = np.asarray(label_based_macro_precision_list).mean()
        # metrics_dict['lbl_macro_recall'] = np.asarray(label_based_macro_recall_list).mean()
        # metrics_dict['lbl_micro_accuracy'] = np.asarray(label_based_micro_accuracy_list).mean()
        # metrics_dict['lbl_micro_precision'] = np.asarray(label_based_micro_precision_list).mean()
        # metrics_dict['lbl_micro_recall'] = np.asarray(label_based_micro_recall_list).mean()
        metrics_dict['f1_score'] = np.asarray(f1_score_list).mean()
        targets = np.concatenate(targets)
        predictions = np.concatenate(predictions)
        return val_loss, targets, predictions


def print_metric_results(metrics: dict, op_file=None):
    print("Exact match ratio: ", metrics['emr'])
    print("1/0 Loss: ", metrics['one_zero'])
    print("Hamming Distance: ", metrics['hamming'])
    print("Accuracy: ", metrics['accuracy'])
    print("Precision: ", metrics['precision'])
    print("Recall: ", metrics['recall'])
    print("F1 Score:", metrics['f1_score'])

    if op_file is not None:
        op_file.write(f"{metrics['emr']},{metrics['one_zero']},{metrics['hamming']},{metrics['accuracy']},{metrics['precision']},{metrics['recall']},{metrics['f1_score']}")


if __name__ == "__main__":
    IMAGE_ROOT = 'dataset_small'
    data_paths = dict()
    dataframes = dict()
    image_paths = dict()
    labels = dict()
    magnifications = ['global', 'local']
    for magnification in magnifications:
        data_paths[magnification] = [IMAGE_ROOT + os.sep + magnification + os.sep + 'images',
                                     IMAGE_ROOT + "_" + magnification + '_labels.csv']
        df = pd.read_csv(data_paths[magnification][1])
        unnamed_col = df.columns[df.columns.str.contains('unnamed', case=False)]
        df.drop(unnamed_col, axis=1, inplace=True)
        dataframes[magnification] = df
        image_paths[magnification] = glob(data_paths.get(magnification)[0] + os.sep
                                          + "*.jpg")
        labels[magnification] = dataframes[magnification].iloc[:, 2:-1].values
        print("Number of " + magnification + " images:")
        print(len(image_paths.get(magnification)))

    # initialize the model
    model = build_model(pretrained=False, fine_tune=True, num_classes=5).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    wce_loss_function = MultiLabelWCELoss(dataframes['local'], device=DEVICE)
    eigen_loss_function = EigenLoss(threshold=0.95)
    ce_loss_function = nn.CrossEntropyLoss()
    mul_margin_loss_function = nn.MultiLabelMarginLoss()
    cosine_loss_function = CosineLoss()
    hamming_loss_function = MultilabelHammingDistance(num_labels=5).to(device=DEVICE)
    ranking_loss_function = MultilabelRankingLoss(num_labels=5).to(device=DEVICE)
    loss_functions = [wce_loss_function, eigen_loss_function, ce_loss_function, mul_margin_loss_function,
                      cosine_loss_function, hamming_loss_function, ranking_loss_function]
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths['local'],
                                                                                      labels['local'], test_size=0.2)
    # train dataset
    train_data = WSIDataset(train_image_paths, train_labels, train=True)
    # validation dataset
    valid_data = WSIDataset(test_image_paths, test_labels, train=False)
    # train data loader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # validation data loader
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

    train_loss = []
    valid_loss = []
    train_metrics = dict()
    # train_metrics['emr'] = list()
    # train_metrics['one_zero'] = list()
    # train_metrics['hamming'] = list()
    # train_metrics['accuracy'] = list()
    # train_metrics['precision'] = list()
    # train_metrics['recall'] = list()
    # train_metrics['f1_score'] = list()
    # train_metrics['ex_accuracy'] = list()
    # train_metrics['ex_precision'] = list()
    # train_metrics['lbl_macro_accuracy'] = list()
    # train_metrics['lbl_macro_precision'] = list()
    # train_metrics['lbl_macro_recall'] = list()
    # train_metrics['lbl_micro_accuracy'] = list()
    # train_metrics['lbl_micro_precision'] = list()
    # train_metrics['lbl_micro_recall'] = list()

    val_metrics = dict()
    # val_metrics['emr'] = list()
    # val_metrics['one_zero'] = list()
    # val_metrics['hamming'] = list()
    # val_metrics['accuracy'] = list()
    # val_metrics['precision'] = list()
    # val_metrics['recall'] = list()
    # val_metrics['ex_accuracy'] = list()
    # val_metrics['ex_precision'] = list()
    # val_metrics['lbl_macro_accuracy'] = list()
    # val_metrics['lbl_macro_precision'] = list()
    # val_metrics['lbl_macro_recall'] = list()
    # val_metrics['lbl_micro_accuracy'] = list()
    # val_metrics['lbl_micro_precision'] = list()
    # val_metrics['lbl_micro_recall'] = list()
    # val_metrics['f1_score'] = list()

    train_file = open("Results/train_results.csv", "w")
    val_file = open("Results/val_results.csv", "w")
    train_file.write("Epoch,Exact_Match_Ratio,One_Zero_Loss,Hamming_Distance,Accuracy,Precision,Recall,F1-Score")
    val_file.write("Epoch,Exact_Match_Ratio,One_Zero_Loss,Hamming_Distance,Accuracy,Precision,Recall,F1-Score")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1} of {EPOCHS}")
        train_epoch_loss, targets, predictions = train(
            model, train_loader, optimizer, loss_functions, train_metrics, DEVICE
        )
        print("Classification Report")
        print(classification_report(targets, predictions))

        print("Training Results")
        print_metric_results(train_metrics, train_file)

        valid_epoch_loss, val_targets, val_predictions = validate(
            model, valid_loader, loss_functions, val_metrics, DEVICE
        )
        print("Validation Results")
        print_metric_results(val_metrics, val_file)

        print("Classification Report")
        print(classification_report(val_targets, val_predictions))
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {valid_epoch_loss:.4f}')
    train_file.close()
    val_file.close()
