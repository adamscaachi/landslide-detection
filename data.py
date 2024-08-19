import glob
import h5py
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class Data:

    def __init__(self, bands, augment=False):
        self.bands = bands
        self.augment = augment
        self.read_and_process_data()
        self.split_data()
        self.train_loader = self.create_loader(self.train_images, self.train_masks, self.augment)
        self.val_loader = self.create_loader(self.val_images, self.val_masks)
        self.test_loader = self.create_loader(self.test_images, self.test_masks)

    def read_and_process_data(self):
        image_paths = glob.glob('data/img/*.h5')
        mask_paths = glob.glob('data/mask/*.h5')    
        num_samples = len(image_paths)
        num_channels = len(self.bands)
        self.images = np.zeros((num_samples, num_channels, 128, 128))
        self.masks = np.zeros((num_samples, 1, 128, 128))
        for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            self.process_image(image_path, i)
            self.process_mask(mask_path, i)

    def process_image(self, image_path, index):
        with h5py.File(image_path, 'r') as f:
            image = np.array(f['img'])
            for i, band in enumerate(self.bands):
                if band.startswith("B") and band[1:].isdigit() and 1 <= int(band[1:]) <= 14:
                    self.images[index, i, :, :] = self.normalise(image[:, :, int(band[1:]) - 1])
                elif band == "NDVI":
                    self.images[index, i, :, :] = self.normalise((image[:, :, 7] - image[:, :, 3]) / (image[:, :, 7] + image[:, :, 3] + 1e-14))
                elif band == "NDWI":
                    self.images[index, i, :, :] = self.normalise((image[:, :, 2] - image[:, :, 7]) / (image[:, :, 2] + image[:, :, 7] + 1e-14))
                else:
                    raise ValueError(f"Band {band} is not available.")

    def process_mask(self, mask_path, index):
        with h5py.File(mask_path, 'r') as f:
            mask = np.array(f['mask'])
            self.masks[index, 0, :, :] = mask

    def normalise(self, data):
        min = np.min(data)
        max = np.max(data)
        return (data - min) / (max - min + 1e-14)

    def split_data(self):
        self.train_images, temp_images, self.train_masks, temp_masks = train_test_split(self.images, self.masks, test_size=128, random_state=42)
        self.val_images, self.test_images, self.val_masks, self.test_masks = train_test_split(temp_images, temp_masks, test_size=64, random_state=42)
        self.cleanup(temp_images, temp_masks)

    def cleanup(self, temp_images, temp_masks):
        del self.images
        del self.masks
        del temp_images
        del temp_masks

    def create_loader(self, images, masks, augment=False): 
        images_tensor = torch.tensor(images, dtype=torch.float32)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)
        dataset = AugmentedDataset(images_tensor, masks_tensor, augment)
        return DataLoader(dataset, batch_size=64, shuffle=True)

class AugmentedDataset(Dataset):

    def __init__(self, images_tensor, masks_tensor, augment):
        self.images = images_tensor
        self.masks = masks_tensor
        self.augment = augment
        self.augmentations = [self.flip, self.dropout]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.augment and random.random() > 0.1:
            image, mask = self.augment_image_and_mask(image, mask)
        return image, mask

    def augment_image_and_mask(self, image, mask):
        for function in self.augmentations:
            if random.random() > 0.5:
                image, mask = function(image, mask)
        return image, mask

    def flip(self, image, mask):
        axis = random.choice([[1], [2], [1, 2]])
        image = torch.flip(image, axis)
        mask = torch.flip(mask, axis)
        return image, mask

    def dropout(self, image, mask):
        dropout_mask = torch.rand_like(image) > 0.001
        image = image * dropout_mask
        return image, mask