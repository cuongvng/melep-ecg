import sys
import torch
torch.manual_seed(0)
import numpy as np
import torch.optim as optim
import os
from datetime import datetime
from data.dataset import MELEPDataset
from utils import classify, get_f1, get_accuracy, load_pretrained_model
from melep import compute_melep

MODELS = [
    'resnet1d101',
    'bilstm'
]

SOURCE_LABELS = {
    'ptbxl': ['NORM', 'MI', 'STTC', 'HYP', 'CD'],
    'shaoxing': [
        'SB', 'NORM', 'AFL', 'STach', 'TAb', 'LVH', 'STC', 'TInv', 'SA', 'AF', 'LQRSV', 'STD', 'LAD', 'PAC', 'PR', 'STT', 'IAVB', 'CRBBB', 'PVC', 'QAb'
        ],
    'cpsc': ['NORM', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE'],
    'georgia': ['NORM', 'AF', 'IAVB', 'SB', 'LAD', 'STach', 'TAb', 'TInv', 'LQT', 'PAC']
}


def experiment(model_name, pretrained_checkpoint, source_data, source_labels, target_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    num_epochs = 50
    batch_size = 256
    data_dir = f'./data/ckp_selection/{target_data}'
    fold = 'ckp'
    log_dir = './ckp_log'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    log_file = os.path.join(log_dir, f'{model_name}_{source_data}_to_{target_data}.csv')
    
    with open(log_file, 'w') as f:
        f.write(f"fold|melep|avg_f1\n")

    train_data = MELEPDataset(fold, "train", data_dir)
    test_data = MELEPDataset(fold, "test", data_dir)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    ## Load fresh pretrained model
    model = load_pretrained_model(pretrained_checkpoint, device)

    ### COMPUTE MELEP (on cpu)
    if model_name == 'bilstm':
        model.device = torch.device("cpu")

    melep_score = compute_melep(model, train_data, source_labels)
    print(f"MELEP = {melep_score:.4f}")

    ### FINETUNE
    # Replace the top layer to fit output

    if model_name == 'resnet1d101':
        model.fc2 = torch.nn.Linear(in_features=512, out_features=len(train_data.CLASSES), bias=True)
        assert model.fc2.weight.size(0) == len(train_data.CLASSES)
        
    else: # bilstm
        seq_len = 1000
        hidden_size = 100
        
        model.fc = torch.nn.Linear(in_features=seq_len*2*hidden_size, out_features=len(train_data.CLASSES), bias=True)
        assert model.fc.weight.size(0) == len(train_data.CLASSES)
        model.device = device

    model.to(device=device, dtype=torch.double)

    sample_weights = train_data.y.sum(axis=0)/len(train_data)
    best = 0.0
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=None)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device, dtype=torch.double)

            optimizer.zero_grad()        
            y_hat = model(data)

            loss = loss_func(y_hat.float(), labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                log = "Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss)
                print(log)

            # Free up GPU memory
            if device == torch.device('cuda'):
                del data, labels, y_hat
                torch.cuda.empty_cache()

        # Eval on test set
        y_trues_val, y_preds_val= classify(model, device, test_data)
        f1_val = get_f1(y_trues_val, y_preds_val)
        f1_mean = np.average(f1_val, weights=sample_weights)
        best = max(best, f1_mean) 

    with open(log_file, 'a') as f:
        f.write(f"{fold}|{melep_score:.4f}|{best:.4f}\n")

    
def target(target_data):
    assert target_data in ['ptbxl', 'shaoxing', 'cpsc', 'georgia']

    for model_name in MODELS:
        for source_data in SOURCE_LABELS.keys():

            if source_data != target_data:
                source_labels = SOURCE_LABELS[source_data]
                pretrained_checkpoint = f"./models/{model_name}_{source_data}.pt"
                experiment(model_name, pretrained_checkpoint, source_data, source_labels, target_data)


if __name__ == '__main__':
    for target_data in ['ptbxl', 'cpsc', 'georgia', 'shaoxing']:
        target(target_data)