import glob
import h5py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class Data:

    def __init__(self, bands):
        self.bands = bands
        self.read_and_process_data()
        self.split_data()
        self.train_loader = self.create_loader(self.train_images, self.train_masks)
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

    def create_loader(self, images, masks):
        images_tensor = torch.tensor(images, dtype=torch.float32)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)
        dataset_tensor = TensorDataset(images_tensor, masks_tensor)
        return DataLoader(dataset_tensor, batch_size=64, shuffle=True)