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


def experiment_shaoxing(model_name, pretrained_checkpoint, source_data, source_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    num_epochs = 50
    batch_size = 256
    data_dir = './data/shaoxing_ningbo/fold'

    log_dir = './log'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    fold_log = log_dir + f'/fold_log/{model_name}_{source_data}_to_shaoxing'
    if not os.path.isdir(fold_log):
        os.mkdir(fold_log)

    log_file = os.path.join(log_dir, f'{model_name}_{source_data}_to_shaoxing_{datetime.now().isoformat()}.csv')
    
    with open(log_file, 'w') as f:
        f.write(f"fold|melep|f1|acc\n")

    for fold in range(100):

        finetune_log = os.path.join(fold_log, f'{model_name}_{source_data}_to_shaoxing_fold_{fold}_{datetime.now().isoformat()}.csv')
        with open(finetune_log, 'w') as f:
            f.write(f"epoch|f1|acc\n")

        print(f'\nFOLD {fold}:')
        train_data = MELEPDataset(fold, "train", data_dir)
        test_data = MELEPDataset(fold, "test", data_dir)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )

        ## Load fresh pretrained model
        model = load_pretrained_model(pretrained_checkpoint, device)

        ### COMPUTE MELEP
        melep_score = compute_melep(model, train_data, source_labels)
        print(f"MELEP = {melep_score:.4f}")

        ### FINETUNE
        # Replace the top layer to fit output
        model.fc2 = torch.nn.Linear(in_features=512, out_features=len(train_data.CLASSES), bias=True)
        assert model.fc2.weight.size(0) == len(train_data.CLASSES)
        model.to(device=device, dtype=torch.double)

        sample_weights = train_data.y.sum(axis=0)/len(train_data)
        best_f1 = [0.0] * len(sample_weights)
        best_acc = [0.0] * len(sample_weights)

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

                if batch_idx % 50 == 0:
                    log = "Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss)
                    print(log)

                # Free up GPU memory
                if device == torch.device('cuda'):
                    del data, labels, y_hat
                    torch.cuda.empty_cache()

            # Eval on test set
            y_trues_val, y_preds_val= classify(model, device, test_data)
            f1_val = get_f1(y_trues_val, y_preds_val)
            acc = get_accuracy(y_trues_val, y_preds_val)

            with open(finetune_log, 'a') as f:
                f.write(f"{epoch}|{list(f1_val.round(4))}|{list(acc.round(4))}\n")

            f1_mean = np.average(f1_val, weights=sample_weights)
            if  f1_mean > np.average(best_f1, weights=sample_weights):
                best_f1 = f1_val
                print("Best f1:", f1_mean)
            
            if np.average(acc, weights=sample_weights) > np.average(best_acc, weights=sample_weights):
                best_acc = acc

        with open(log_file, 'a') as f:
            f.write(f"{fold}|{melep_score:.4f}|{list(best_f1.round(4))}|{list(best_acc.round(4))}\n")


def main():
    MODELS = [
        'resnet1d101',
    ]
    
    SOURCE_LABELS = {
        'ptbxl': ['NORM', 'MI', 'STTC', 'HYP', 'CD'],
        'cpsc': ['NORM', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'VEB', 'STD', 'STE'],
        'georgia': ['NORM', 'AF', 'IAVB', 'SB', 'LAD', 'STach', 'TAb', 'TInv', 'LQT', 'PAC']
    }


    for model_name in MODELS:
        for source_data in SOURCE_LABELS.keys():
            source_labels = SOURCE_LABELS[source_data]
            pretrained_checkpoint = f"./models/{model_name}_{source_data}.pt"

            experiment_shaoxing(model_name, pretrained_checkpoint, source_data, source_labels)


if __name__ == '__main__':
    main()