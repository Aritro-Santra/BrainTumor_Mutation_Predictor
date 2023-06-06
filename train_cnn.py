import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


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
def train(model, dataloader, optimizer, criterion, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    targets = []
    predictions = []
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
        # print(binarized_outputs.requires_grad)
        loss = criterion(target, binarized_outputs)
        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        y_true = target.detach().cpu().numpy()
        y_pred = binarized_outputs.detach().cpu().numpy()
        # confustion_matrix = multilabel_confusion_matrix(y_true, y_pred)
        # print(confustion_matrix)
        # print(classification_report(y_true, y_pred))
        targets.append(y_true)
        predictions.append(y_pred)
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    train_loss = train_running_loss / counter
    return train_loss, targets, predictions


# validation function
def validate(model, dataloader, criterion, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    targets = []
    predictions = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset) / dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            binarized_outputs = torch.round(outputs)
            loss = criterion(target, binarized_outputs)
            val_running_loss += loss.item()
            y_true = target.detach().cpu().numpy()
            y_pred = binarized_outputs.detach().cpu().numpy()
            targets.append(y_true)
            predictions.append(y_pred)
        val_loss = val_running_loss / counter
        targets = np.concatenate(targets)
        predictions = np.concatenate(predictions)
        return val_loss, targets, predictions


DATA_ROOT = 'dataset_small'
data_paths = dict()
dataframes = dict()
image_paths = dict()
labels = dict()
magnifications = ['global', 'local']
for magnification in magnifications:
    data_paths[magnification] = [DATA_ROOT + os.sep + magnification + os.sep + 'images',
                                 DATA_ROOT + os.sep + magnification + os.sep +
                                 magnification + '_labels.csv']
    df = pd.read_csv(data_paths[magnification][1])
    unnamed_col = df.columns[df.columns.str.contains('unnamed', case=False)]
    df.drop(unnamed_col, axis=1, inplace=True)
    dataframes[magnification] = df
    image_paths[magnification] = glob(data_paths.get(magnification)[0] + os.sep
                                      + "*.jpg")
    labels[magnification] = dataframes[magnification].iloc[:, 2:-1].values
    print("Number of " + magnification + " images:")
    print(len(image_paths.get(magnification)))


# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = build_model(pretrained=False, fine_tune=True, num_classes=5).to(device)
# learning parameters
lr = 1e-3
epochs = 30
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.BCELoss()

train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths['local'], labels['local'], test_size=0.2)
# train dataset
train_data = WSIDataset(train_image_paths, train_labels, train=True)
# validation dataset
valid_data = WSIDataset(test_image_paths, test_labels, train=False)
# train data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# validation data loader
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
