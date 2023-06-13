import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import warnings
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from WSIDataset import WSIDataset
from LossFunction import MultiLabelWCELoss, EigenLoss
from Metrics import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

warnings.filterwarnings("ignore")

# initialize the computation device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# learning parameters
LR = 1e-3
EPOCHS = 30
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
def train(model, dataloader, optimizer, criterions, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    targets = []
    predictions = []
    emr_list = []
    one_zero_loss_list = []
    hamming_loss_list = []
    example_based_accuracy_list = []
    example_based_precision_list = []
    label_based_macro_accuracy_list = []
    label_based_macro_precision_list = []
    label_based_macro_recall_list = []
    label_based_micro_accuracy_list = []
    label_based_micro_precision_list = []
    label_based_micro_recall_list = []
    f1_score_list = []
    metrics_dict = dict()
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
        total_loss = torch.add(wce_loss, eigen_loss)
        train_running_loss += total_loss.item()
        # backpropagation
        total_loss.backward()
        # update optimizer parameters
        optimizer.step()
        y_true = target.detach().cpu()
        y_pred = binarized_outputs.detach().cpu()

        emr_score = emr(y_true, y_pred)
        one_zero_loss_score = one_zero_loss(y_true, y_pred)
        hamming_loss_score = hamming_loss(y_true, y_pred)
        example_based_accuracy_score = example_based_accuracy(y_true, y_pred)
        example_based_precision_score = example_based_precision(y_true, y_pred)
        label_based_macro_accuracy_score = label_based_macro_accuracy(y_true, y_pred)
        label_based_macro_precision_score = label_based_macro_precision(y_true, y_pred)
        label_based_macro_recall_score = label_based_macro_recall(y_true, y_pred)
        label_based_micro_accuracy_score = label_based_micro_accuracy(y_true, y_pred)
        label_based_micro_precision_score = label_based_micro_precision(y_true, y_pred)
        label_based_micro_recall_score = label_based_micro_recall(y_true, y_pred)
        f1_score_val = f1_score(y_true, y_pred)
        emr_list.append(emr_score)
        one_zero_loss_list.append(one_zero_loss_score)
        hamming_loss_list.append(hamming_loss_score)
        example_based_accuracy_list.append(example_based_accuracy_score)
        example_based_precision_list.append(example_based_precision_score)
        label_based_macro_accuracy_list.append(label_based_macro_accuracy_score)
        label_based_macro_precision_list.append(label_based_macro_precision_score)
        label_based_macro_recall_list.append(label_based_macro_recall_score)
        label_based_micro_accuracy_list.append(label_based_micro_accuracy_score)
        label_based_micro_precision_list.append(label_based_micro_precision_score)
        label_based_micro_recall_list.append(label_based_micro_recall_score)
        f1_score_list.append(f1_score_val)

        # confustion_matrix = multilabel_confusion_matrix(y_true, y_pred)
        # print(confustion_matrix)
        # print(classification_report(y_true, y_pred))
        targets.append(y_true.numpy())
        predictions.append(y_pred.numpy())
    metrics_dict['emr'] = np.asarray(emr_list).mean()
    metrics_dict['one_zero'] = np.asarray(one_zero_loss_list).mean()
    metrics_dict['hamming'] = np.asarray(hamming_loss_list).mean()
    metrics_dict['ex_accuracy'] = np.asarray(example_based_accuracy_list).mean()
    metrics_dict['ex_precision'] = np.asarray(example_based_precision_list).mean()
    metrics_dict['lbl_macro_accuracy'] = np.asarray(label_based_macro_accuracy_list).mean()
    metrics_dict['lbl_macro_precision'] = np.asarray(label_based_macro_precision_list).mean()
    metrics_dict['lbl_macro_recall'] = np.asarray(label_based_macro_recall_list).mean()
    metrics_dict['lbl_micro_accuracy'] = np.asarray(label_based_micro_accuracy_list).mean()
    metrics_dict['lbl_micro_precision'] = np.asarray(label_based_micro_precision_list).mean()
    metrics_dict['lbl_micro_recall'] = np.asarray(label_based_micro_recall_list).mean()
    metrics_dict['f1_score'] = np.asarray(f1_score_list).mean()
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    train_loss = train_running_loss / counter
    return train_loss, targets, predictions, metrics_dict


# validation function
def validate(model, dataloader, criterions, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    targets = []
    predictions = []
    emr_list = []
    one_zero_loss_list = []
    hamming_loss_list = []
    example_based_accuracy_list = []
    example_based_precision_list = []
    label_based_macro_accuracy_list = []
    label_based_macro_precision_list = []
    label_based_macro_recall_list = []
    label_based_micro_accuracy_list = []
    label_based_micro_precision_list = []
    label_based_micro_recall_list = []
    f1_score_list = []
    metrics_dict = dict()
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
            hamming_loss_score = hamming_loss(y_true, y_pred)
            example_based_accuracy_score = example_based_accuracy(y_true, y_pred)
            example_based_precision_score = example_based_precision(y_true, y_pred)
            label_based_macro_accuracy_score = label_based_macro_accuracy(y_true, y_pred)
            label_based_macro_precision_score = label_based_macro_precision(y_true, y_pred)
            label_based_macro_recall_score = label_based_macro_recall(y_true, y_pred)
            label_based_micro_accuracy_score = label_based_micro_accuracy(y_true, y_pred)
            label_based_micro_precision_score = label_based_micro_precision(y_true, y_pred)
            label_based_micro_recall_score = label_based_micro_recall(y_true, y_pred)
            f1_score_val = f1_score(y_true, y_pred)
            emr_list.append(emr_score)
            one_zero_loss_list.append(one_zero_loss_score)
            hamming_loss_list.append(hamming_loss_score)
            example_based_accuracy_list.append(example_based_accuracy_score)
            example_based_precision_list.append(example_based_precision_score)
            label_based_macro_accuracy_list.append(label_based_macro_accuracy_score)
            label_based_macro_precision_list.append(label_based_macro_precision_score)
            label_based_macro_recall_list.append(label_based_macro_recall_score)
            label_based_micro_accuracy_list.append(label_based_micro_accuracy_score)
            label_based_micro_precision_list.append(label_based_micro_precision_score)
            label_based_micro_recall_list.append(label_based_micro_recall_score)
            f1_score_list.append(f1_score_val)
            targets.append(y_true.numpy())
            predictions.append(y_pred.numpy())
        val_loss = val_running_loss / counter
        metrics_dict['emr'] = np.asarray(emr_list).mean()
        metrics_dict['one_zero'] = np.asarray(one_zero_loss_list).mean()
        metrics_dict['hamming'] = np.asarray(hamming_loss_list).mean()
        metrics_dict['ex_accuracy'] = np.asarray(example_based_accuracy_list).mean()
        metrics_dict['ex_precision'] = np.asarray(example_based_precision_list).mean()
        metrics_dict['lbl_macro_accuracy'] = np.asarray(label_based_macro_accuracy_list).mean()
        metrics_dict['lbl_macro_precision'] = np.asarray(label_based_macro_precision_list).mean()
        metrics_dict['lbl_macro_recall'] = np.asarray(label_based_macro_recall_list).mean()
        metrics_dict['lbl_micro_accuracy'] = np.asarray(label_based_micro_accuracy_list).mean()
        metrics_dict['lbl_micro_precision'] = np.asarray(label_based_micro_precision_list).mean()
        metrics_dict['lbl_micro_recall'] = np.asarray(label_based_micro_recall_list).mean()
        metrics_dict['f1_score'] = np.asarray(f1_score_list).mean()
        targets = np.concatenate(targets)
        predictions = np.concatenate(predictions)
        return val_loss, targets, predictions, metrics_dict


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
    wce_loss_function = MultiLabelWCELoss(dataframes['global'], device=DEVICE)
    eigen_loss_function = EigenLoss(threshold=0.95)

    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths['global'],
                                                                                      labels['global'], test_size=0.2)
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
    train_metrics['emr'] = list()
    train_metrics['one_zero'] = list()
    train_metrics['hamming'] = list()
    train_metrics['ex_accuracy'] = list()
    train_metrics['ex_precision'] = list()
    train_metrics['lbl_macro_accuracy'] = list()
    train_metrics['lbl_macro_precision'] = list()
    train_metrics['lbl_macro_recall'] = list()
    train_metrics['lbl_micro_accuracy'] = list()
    train_metrics['lbl_micro_precision'] = list()
    train_metrics['lbl_micro_recall'] = list()
    train_metrics['f1_score'] = list()

    val_metrics = dict()
    val_metrics['emr'] = list()
    val_metrics['one_zero'] = list()
    val_metrics['hamming'] = list()
    val_metrics['ex_accuracy'] = list()
    val_metrics['ex_precision'] = list()
    val_metrics['lbl_macro_accuracy'] = list()
    val_metrics['lbl_macro_precision'] = list()
    val_metrics['lbl_macro_recall'] = list()
    val_metrics['lbl_micro_accuracy'] = list()
    val_metrics['lbl_micro_precision'] = list()
    val_metrics['lbl_micro_recall'] = list()
    val_metrics['f1_score'] = list()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1} of {EPOCHS}")
        print("Training...")
        train_epoch_loss, targets, predictions, train_metrics_values = train(
            model, train_loader, optimizer, [wce_loss_function, eigen_loss_function], DEVICE
        )
        print("Classification Report")
        print(classification_report(targets, predictions))

        print("Exact match ratio: ", train_metrics_values['emr'])
        print("1/0 Loss: ", train_metrics_values['one_zero'])
        print("Hamming Loss: ", train_metrics_values['hamming'])
        print("Example based Accuracy: ", train_metrics_values['ex_accuracy'])
        print("Example based precision: ", train_metrics_values['ex_precision'])
        print("Macro Averaged Accuracy: ", train_metrics_values['lbl_macro_accuracy'])
        print("Macro Averaged Precision: ", train_metrics_values['lbl_macro_precision'])
        print("Macro Averaged Recall: ", train_metrics_values['lbl_macro_recall'])
        print("Micro Averaged Accuracy: ", train_metrics_values['lbl_micro_accuracy'])
        print("Micro Averaged Precision: ", train_metrics_values['lbl_micro_precision'])
        print("Micro Averaged Recall: ", train_metrics_values['lbl_micro_recall'])
        print("F1 Score:", train_metrics_values['f1_score'])

        train_metrics['emr'].append(train_metrics_values['emr'])
        train_metrics['one_zero'].append(train_metrics_values['one_zero'])
        train_metrics['hamming'].append(train_metrics_values['hamming'])
        train_metrics['ex_accuracy'].append(train_metrics_values['ex_accuracy'])
        train_metrics['ex_precision'].append(train_metrics_values['ex_precision'])
        train_metrics['lbl_macro_accuracy'].append(train_metrics_values['lbl_macro_accuracy'])
        train_metrics['lbl_macro_precision'].append(train_metrics_values['lbl_macro_precision'])
        train_metrics['lbl_macro_recall'].append(train_metrics_values['lbl_macro_recall'])
        train_metrics['lbl_micro_accuracy'].append(train_metrics_values['lbl_micro_accuracy'])
        train_metrics['lbl_micro_precision'].append(train_metrics_values['lbl_micro_precision'])
        train_metrics['lbl_micro_recall'].append(train_metrics_values['lbl_micro_recall'])
        train_metrics['f1_score'].append(train_metrics_values['f1_score'])

        print("Validating...")
        valid_epoch_loss, val_targets, val_predictions, val_metrics_values = validate(
            model, valid_loader, [wce_loss_function, eigen_loss_function], DEVICE
        )

        print("Exact match ratio: ", val_metrics_values['emr'])
        print("1/0 Loss: ", val_metrics_values['one_zero'])
        print("Hamming Loss: ", val_metrics_values['hamming'])
        print("Example based Accuracy: ", val_metrics_values['ex_accuracy'])
        print("Example based precision: ", val_metrics_values['ex_precision'])
        print("Macro Averaged Accuracy: ", val_metrics_values['lbl_macro_accuracy'])
        print("Macro Averaged Precision: ", val_metrics_values['lbl_macro_precision'])
        print("Macro Averaged Recall: ", val_metrics_values['lbl_macro_recall'])
        print("Micro Averaged Accuracy: ", val_metrics_values['lbl_micro_accuracy'])
        print("Micro Averaged Precision: ", val_metrics_values['lbl_micro_precision'])
        print("Micro Averaged Recall: ", val_metrics_values['lbl_micro_recall'])
        print("F1 Score:", val_metrics_values['f1_score'])

        val_metrics['emr'].append(val_metrics_values['emr'])
        val_metrics['one_zero'].append(val_metrics_values['one_zero'])
        val_metrics['hamming'].append(val_metrics_values['hamming'])
        val_metrics['ex_accuracy'].append(val_metrics_values['ex_accuracy'])
        val_metrics['ex_precision'].append(val_metrics_values['ex_precision'])
        val_metrics['lbl_macro_accuracy'].append(val_metrics_values['lbl_macro_accuracy'])
        val_metrics['lbl_macro_precision'].append(val_metrics_values['lbl_macro_precision'])
        val_metrics['lbl_macro_recall'].append(val_metrics_values['lbl_macro_recall'])
        val_metrics['lbl_micro_accuracy'].append(val_metrics_values['lbl_micro_accuracy'])
        val_metrics['lbl_micro_precision'].append(val_metrics_values['lbl_micro_precision'])
        val_metrics['lbl_micro_recall'].append(val_metrics_values['lbl_micro_recall'])
        val_metrics['f1_score'].append(val_metrics_values['f1_score'])

        print("Classification Report")
        print(classification_report(val_targets, val_predictions))
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {valid_epoch_loss:.4f}')
