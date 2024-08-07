import glob
import h5py
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class Data:

    def __init__(self):
        self.read_and_process_data()
        self.split_data()
        self.train_loader = self.create_loader(self.train_images, self.train_masks)
        self.val_loader = self.create_loader(self.val_images, self.val_masks)

    def read_and_process_data(self):
        image_paths = glob.glob('data/train/img/*.h5')
        mask_paths = glob.glob('data/train/mask/*.h5')    
        num_samples = len(image_paths)
        self.images = np.zeros((num_samples, 6, 128, 128))
        self.masks = np.zeros((num_samples, 1, 128, 128))
        for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            self.process_image(image_path, i)
            self.process_mask(mask_path, i)

    def process_image(self, image_path, index):
        with h5py.File(image_path, 'r') as f:
            image = np.array(f['img'])
            image_red = image[:, :, 3]
            image_nir = image[:, :, 7]
            self.images[index, 0, :, :] = self.normalise(image_red)
            self.images[index, 1, :, :] = self.normalise(image[:, :, 2]) 
            self.images[index, 2, :, :] = self.normalise(image[:, :, 1]) 
            self.images[index, 3, :, :] = self.normalise((image_nir - image_red) / (image_nir + image_red + 1e-14))
            self.images[index, 4, :, :] = self.normalise(image[:, :, 12]) 
            self.images[index, 5, :, :] = self.normalise(image[:, :, 13])

    def process_mask(self, mask_path, index):
        with h5py.File(mask_path, 'r') as f:
            mask = np.array(f['mask'])
            self.masks[index, 0, :, :] = mask

    def normalise(self, data):
        min = np.min(data)
        max = np.max(data)
        return (data - min) / (max - min + 1e-14)

    def split_data(self):
        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(self.images, self.masks, test_size=0.2, random_state=42)

    def create_loader(self, images, masks):
        images_tensor = torch.tensor(images, dtype=torch.float32)
        masks_tensor = torch.tensor(masks, dtype=torch.float32)
        dataset_tensor = TensorDataset(images_tensor, masks_tensor)
        return DataLoader(dataset_tensor, batch_size=64, shuffle=True)