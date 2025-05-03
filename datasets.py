import pickle
import torch
import numpy as np
import random
import torchvision

def unpickle(f):
    with open(f, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def new_random_crop(labels, new_crop_size):
    crop_map = create_crop_map(len(labels), new_crop_size)
    good_indices = find_good_samples(labels, crop_map, new_crop_size)
    return crop_map, good_indices


def create_crop_map(data_size, crop_size):
    # The crop map assigns an x_shift and y_shift to each sample in the main 64 x 64 dataset.
    crop_map = []
    for i in range(data_size):
        x_shift = random.randint(0, 64 - crop_size)
        y_shift = random.randint(0, 64 - crop_size)
        crop_map.append((x_shift, y_shift))
    return np.array(crop_map)


def get_cropped_sample(index, crop_map, crop_size, data, labels):
    x_shift, y_shift = crop_map[index]
    cropped_features = data[index, :, x_shift : x_shift + crop_size, y_shift : y_shift + crop_size]
    cropped_label = labels[index, x_shift : x_shift + crop_size, y_shift : y_shift + crop_size]
        
    return cropped_features, cropped_label


def find_good_samples(labels, crop_map, crop_size):
    # Finds the indices of samples that have no missing data in their labels.
    # This is determined AFTER generating a crop map and applying the crop to the original 64 x 64 label.
    good_indices = []
    for i in range(len(labels)):
        x_shift, y_shift = crop_map[i]
        if np.all(np.invert(labels[i, x_shift : x_shift + crop_size, y_shift : y_shift + crop_size] == -1)):
            good_indices.append(i)
    return np.array(good_indices)


class WildfireDataset(torch.utils.data.Dataset):
    def __init__(self, data_filename, labels_filename, features=None, crop_size=64):
        self.data, self.labels = unpickle(data_filename), unpickle(labels_filename)
        self.crop_size = crop_size

        random.seed(1)
        self.crop_map, self.good_indices = new_random_crop(self.labels, self.crop_size)

        if features:
            assert isinstance(features, list)
        self.features = sorted(features) if features else None
        
        print(f"data size: {self.data.nbytes}")
        print(f"label size: {self.labels.nbytes}")
        print(f"crop_map size: {self.crop_map.nbytes}")
        print(f"good_indices size: {self.good_indices.nbytes}")
        print(f"total size: {self.data.nbytes + self.labels.nbytes + self.crop_map.nbytes + self.good_indices.nbytes}")
        print("finished initializing WildfireDataset")
        
    def __len__(self):
        return len(self.good_indices)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        index = self.good_indices[index]
        
        cropped_features, cropped_label = get_cropped_sample(index, self.crop_map, self.crop_size, self.data, self.labels)

        # Only keep specific features
        if self.features:
            cropped_features = cropped_features[self.features, :, :]

        sample = (torch.from_numpy(cropped_features), torch.from_numpy(np.expand_dims(cropped_label, axis=0)))

        return sample


class RotatedWildfireDataset(torch.utils.data.Dataset):
    # This dataset probably doesn't work if you use the wind direction feature
    def __init__(self, data_filename, labels_filename, features=None, crop_size=64):
        self.data, self.labels = unpickle(data_filename), unpickle(labels_filename)
        self.crop_size = crop_size

        random.seed(1)
        self.crop_map, self.good_indices = new_random_crop(self.labels, self.crop_size)
        
        if features:
            assert isinstance(features, list)
        self.features = sorted(features) if features else None
        
        #self.oversample_indices = self._find_samples_for_oversampling()
        
        print(f"data size: {self.data.nbytes}")
        print(f"label size: {self.labels.nbytes}")
        print(f"crop_map size: {self.crop_map.nbytes}")
        print(f"good_indices size: {self.good_indices.nbytes}")
        print(f"total size: {self.data.nbytes + self.labels.nbytes + self.crop_map.nbytes + self.good_indices.nbytes}")
        print("finished initializing RotatedWildfireDataset")

    def __len__(self):
        return len(self.good_indices) * 4

    def __getitem__(self, index):
        rotation_index = index // len(self.good_indices)
        
        #index = self.good_indices[index % len(self.good_indices)] if index < len(self.good_indices) * 2 else self._get_random_oversample_index()
        index = self.good_indices[index % len(self.good_indices)]
            
        cropped_features, cropped_label = get_cropped_sample(index, self.crop_map, self.crop_size, self.data, self.labels)

        # Only keep specific features
        if self.features:
            cropped_features = cropped_features[self.features, :, :]

        # Perform rotation
        rotations = [0, 90, 180, 270]
        rotation = rotations[rotation_index]
        cropped_features = torchvision.transforms.functional.rotate(torch.from_numpy(cropped_features), rotation)
        cropped_label = torchvision.transforms.functional.rotate(torch.from_numpy(np.expand_dims(cropped_label, axis=0)), rotation)

        sample = (cropped_features, cropped_label)
        
        return sample


class OversampledWildfireDataset(torch.utils.data.Dataset):
    def __init__(self, data_filename, labels_filename, features=None):
        self.data, self.labels = unpickle(data_filename), unpickle(labels_filename)
        self.crop_size = 64

        random.seed(1)
        self.crop_map, self.good_indices = new_random_crop(self.labels, self.crop_size)

        if features:
            assert isinstance(features, list)
        self.features = sorted(features) if features else None
        
        self.oversample_indices = self._find_samples_for_oversampling()
        
        print(f"data size: {self.data.nbytes}")
        print(f"label size: {self.labels.nbytes}")
        print(f"crop_map size: {self.crop_map.nbytes}")
        print(f"good_indices size: {self.good_indices.nbytes}")
        print(f"total size: {self.data.nbytes + self.labels.nbytes + self.crop_map.nbytes + self.good_indices.nbytes}")
        print("finished initializing OversampledWildfireDataset")

    def __len__(self):
        return len(self.good_indices) * 2

    def __getitem__(self, index):
        
        index = self.good_indices[index % len(self.good_indices)] if index < len(self.good_indices) else self._get_random_oversample_index()
            
        cropped_features, cropped_label = get_cropped_sample(index, self.crop_map, self.crop_size, self.data, self.labels)

        # Only keep specific features
        if self.features:
            cropped_features = cropped_features[self.features, :, :]

        sample = (cropped_features, cropped_label)
        
        return sample

    def _find_samples_for_oversampling(self):
        oversample_indices = []
        threshold = 0.05 # Desired percentage of fire pixels in the target fire masks
        
        for i in range(len(self.good_indices)):
            index = self.good_indices[i]
            x_shift, y_shift = self.crop_map[index]
            cropped_label = self.labels[index, x_shift : x_shift + self.crop_size, y_shift : y_shift + self.crop_size]
            unique_target, counts_target = np.unique(cropped_label, return_counts=True)
            target_counts_map = {int(unique_target[j]): int(counts_target[j]) for j in range(len(unique_target))}
            
            for key, value in target_counts_map.items():
                if key == 1 and (value/1024) >= threshold:
                    oversample_indices.append(index)
        return oversample_indices
        
    
    def _get_random_oversample_index(self):
        return random.choice(self.oversample_indices)






























    
