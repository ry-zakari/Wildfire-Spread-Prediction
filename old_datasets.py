import pickle
import torch
import numpy as np
import random

def unpickle(f):
    with open(f, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# Used for testing and validation datasets
class WildfireDataset(torch.utils.data.Dataset):
    def __init__(self, data_filename, labels_filename, transform=None):
        self.data = unpickle(data_filename)
        self.labels = unpickle(labels_filename)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (torch.from_numpy(self.data[idx]), torch.from_numpy(np.expand_dims(self.labels[idx], axis=0)))

        return sample

# Used for the training dataset
class OversampledWildfireDataset(torch.utils.data.Dataset):
    def __init__(self, data_filename, labels_filename, transform=None):
        self.data = unpickle(data_filename)
        self.labels = unpickle(labels_filename)
        self.oversample_indices = []
        
        for i in range(len(self.data)):
            unique_target, counts_target = np.unique(self.labels[i], return_counts=True)
            target_counts_map = {int(unique_target[i]): int(counts_target[i]) for i in range(len(unique_target))}
            
            for key, value in target_counts_map.items():
                if key == 1 and (value/1024) >= 0.15: # adjust the fraction to see which masks have a lot of fire
                    self.oversample_indices.append(i)

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        idx = idx if idx < len(self.data)//2 else self._get_random_oversample_index()

        sample = (torch.from_numpy(self.data[idx]), torch.from_numpy(np.expand_dims(self.labels[idx], axis=0)))
        
        return sample
    
    def _get_random_oversample_index(self):
        return random.choice(self.oversample_indices)
    