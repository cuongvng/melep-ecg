from torch.utils.data import Dataset
import numpy as np
import joblib
import os
import ast

class ShaoxingNingboDataset(Dataset):      
    def __init__(self, fold, purpose, data_dir='./shaoxing_ningbo/fold'):
        assert purpose in ['train', 'test']

        with open(os.path.join(data_dir, f"fold_{fold}.txt"), 'r') as f:
            data = f.readlines()        
        
        self.CLASSES = ast.literal_eval(data[0])
        self.weights = ast.literal_eval(data[1])

        x_dir = os.path.join(data_dir, f'X_{fold}_{purpose}.joblib')
        y_dir = os.path.join(data_dir, f'y_{fold}_{purpose}.joblib')

        self.X = joblib.load(x_dir)
        self.y = joblib.load(y_dir)
                
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index: int):
        # labels: One-hot values
        labels = self.y[index]

        # signals: 2-D array of size (1000, 12) representing 12-lead signals of length 1000
        signals = self.X[index]
        
        return (signals, labels)
