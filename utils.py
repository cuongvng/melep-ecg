import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def load_pretrained_model(checkpoint_dir, device):
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def classify(model, device, dataset, batch_size=128):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    y_trues = np.empty((0, len(dataset.CLASSES)))
    y_preds = np.empty((0, len(dataset.CLASSES)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            y_pred = torch.sigmoid(y_hat).cpu().numpy().round()
            y = y.cpu()

            y_preds = np.concatenate((y_preds, y_pred), axis=0)
            y_trues = np.concatenate((y_trues, y), axis=0)

    return y_trues, y_preds

def get_f1(y_trues, y_preds):
    f1 = []
    for j in range(y_trues.shape[1]):
        f1.append(f1_score(y_trues[:, j], y_preds[:, j]))
    return np.array(f1)

def get_accuracy(y_trues, y_preds):
    acc = []
    for j in range(y_trues.shape[1]):
        acc.append(accuracy_score(y_trues[:, j], y_preds[:, j]))
    return np.array(acc)
