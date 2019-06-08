import torch
import numpy as np
from torch.utils.data import Dataset



class dataset(Dataset):
    def __init__(self, diagnosis='diagnosis.npy', diagnosis_appr_file='diagnosis_appr.npy',
                 infection_file='infection.npy', alpha_file='alpha.npy', ratio=0.8, train=True):
        diagnosis_appr = np.load(diagnosis_appr_file)
        infection = np.load(infection_file)
        diagnosis = np.load(diagnosis)
        alpha = np.load(alpha_file)
        sample_num = diagnosis_appr.shape[0]
        train_number = int(ratio * sample_num)
        self.train_data = {}
        self.train_data['G0'] = infection[:train_number, 0]
        self.train_data['infection'] = infection[:train_number, :]
        self.train_data['diagnosis'] = diagnosis[:train_number, :]

        self.test_data = {}
        self.test_data['G0'] = infection[train_number:, 0]
        self.test_data['diagnosis'] = diagnosis[train_number:, :]
        self.test_data['infection'] = infection[train_number:, :]

        self.train = train
        self.train_number = train_number
        self.alpha = alpha

    def __getitem__(self, index):
        sample = {}
        if self.train:
            data = self.train_data
        else:
            data = self.test_data
            index = index

        sample['G0'] = data['G0'][index]
        sample['diagnosis'] = torch.tensor(data['diagnosis'][index, :])
        sample['infection'] = torch.tensor(data['infection'][index, :])
        return sample

    def __len__(self):
        if self.train:
            return len(self.train_data['G0'])
        else:
            return len(self.test_data['G0'])
