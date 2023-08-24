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
def train(model, dataloader, optimizer, criterion, metrics_dict, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    targets = []
    predictions = []
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
        ce_loss = criterion(target, binarized_outputs)
        train_running_loss += ce_loss.item()

        # backpropagation
        ce_loss.backward()

        # update optimizer parameters
        optimizer.step()
        y_true = target.detach().cpu()
        y_pred = binarized_outputs.detach().cpu()

        accuracy_score = multi_label_accuracy(y_true, y_pred)
        precision_score = multi_label_precision(y_true, y_pred)
        recall_score = multi_label_recall(y_true, y_pred)
        f1_score_val = f1_score(y_true, y_pred)
        accuracy_list.append(accuracy_score)
        precision_list.append(precision_score)
        recall_list.append(recall_score)
        f1_score_list.append(f1_score_val)

        targets.append(y_true.numpy())
        predictions.append(y_pred.numpy())
    metrics_dict['accuracy'] = np.asarray(accuracy_list).mean()
    metrics_dict['precision'] = np.asarray(precision_list).mean()
    metrics_dict['recall'] = np.asarray(recall_list).mean()
    metrics_dict['f1_score'] = np.asarray(f1_score_list).mean()
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    train_loss = train_running_loss / counter
    return train_loss, targets, predictions


# validation function
def validate(model, dataloader, criterion, metrics_dict, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    targets = []
    predictions = []
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
            ce_loss = criterion(target, binarized_outputs)
            val_running_loss += ce_loss.item()
            y_true = target.detach().cpu()
            y_pred = binarized_outputs.detach().cpu()
            accuracy_score = multi_label_accuracy(y_true, y_pred)
            precision_score = multi_label_precision(y_true, y_pred)
            recall_score = multi_label_recall(y_true, y_pred)
            f1_score_val = f1_score(y_true, y_pred)

            accuracy_list.append(accuracy_score)
            precision_list.append(precision_score)
            recall_list.append(recall_score)
            f1_score_list.append(f1_score_val)
            targets.append(y_true.numpy())
            predictions.append(y_pred.numpy())
        val_loss = val_running_loss / counter
        metrics_dict['accuracy'] = np.asarray(accuracy_list).mean()
        metrics_dict['precision'] = np.asarray(precision_list).mean()
        metrics_dict['recall'] = np.asarray(recall_list).mean()
        metrics_dict['f1_score'] = np.asarray(f1_score_list).mean()
        targets = np.concatenate(targets)
        predictions = np.concatenate(predictions)
        return val_loss, targets, predictions


def print_metric_results(metrics: dict, op_file=None):
    print("Accuracy: ", metrics['accuracy'])
    print("Precision: ", metrics['precision'])
    print("Recall: ", metrics['recall'])
    print("F1 Score:", metrics['f1_score'])

    if op_file is not None:
        op_file.write(f"\n{metrics['accuracy']},{metrics['precision']},{metrics['recall']},{metrics['f1_score']}")


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
        labels[magnification] = dataframes[magnification].iloc[:, 2:4].values
        print("Number of " + magnification + " images:")
        print(len(image_paths.get(magnification)))

    # initialize the model
    model = build_model(pretrained=True, fine_tune=True, num_classes=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_loss_function = nn.CrossEntropyLoss()
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
    val_metrics = dict()
    train_file = open("Results/train_results.csv", "w")
    val_file = open("Results/val_results.csv", "w")
    train_file.write("Epoch,Exact_Match_Ratio,One_Zero_Loss,Hamming_Distance,Accuracy,Precision,Recall,F1-Score")
    val_file.write("Epoch,Exact_Match_Ratio,One_Zero_Loss,Hamming_Distance,Accuracy,Precision,Recall,F1-Score")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1} of {EPOCHS}")
        train_epoch_loss, targets, predictions = train(
            model, train_loader, optimizer, ce_loss_function, train_metrics, DEVICE
        )
        print("Classification Report")
        print(classification_report(targets, predictions))

        print("Training Results")
        print_metric_results(train_metrics, train_file)

        valid_epoch_loss, val_targets, val_predictions = validate(
            model, valid_loader, ce_loss_function, val_metrics, DEVICE
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
